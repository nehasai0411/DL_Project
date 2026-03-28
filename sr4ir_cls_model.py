import os
import os.path as osp
import torch
import warnings

from archs import build_network
from losses import build_loss
from torch.nn.functional import interpolate
from utils.cls import MetricLogger, SmoothedValue, calculate_accuracy
from utils.common import save_on_master, quantize, reduce_across_processes, visualize_image_from_batch, calculate_psnr_batch, calculate_lpips_batch, calculate_niqe_batch

from .base_model import BaseModel

def make_model(opt):
  return SR4IRClassificationModel(opt)

class SR4IRClassificationModel(BaseModel):

 def __init__(self, opt):
    super().__init__(opt)
    
    # define network up
    self.net_up = self.model_to_device(torch.nn.UpsamplingBilinear2d(scale_factor=self.scale), is_trainable=False)
    
    # define network sr
    opt['network_sr']['scale'] = self.scale
    self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
    self.load_network(self.net_sr, name='network_sr', tag='net_sr')
    self.net_sr = self.model_to_device(self.net_sr, is_trainable=True)
    self.print_network(self.net_sr, tag='net_sr')
    
    # define network cls
    self.net_cls = build_network(opt['network_cls'], self.text_logger, task=self.task, tag='net_cls')
    self.load_network(self.net_cls, name='network_cls', tag='net_cls')
    self.net_cls = self.model_to_device(self.net_cls, is_trainable=True)
    self.print_network(self.net_cls, tag='net_cls')

    # ===== LPIPS INIT =====
    import lpips
    self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
    for name, param in self.net_cls.named_parameters():
      if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

 def set_mode(self, mode):
    if mode == 'train':
        self.net_sr.train()
        self.net_cls.train()
    elif mode == 'eval':
        self.net_sr.eval()
        self.net_cls.eval()
    else:
        raise NotImplementedError(f"mode {mode} is not supported")
    
 def init_training_settings(self, data_loader_train):
    self.set_mode(mode='train')
    train_opt = self.opt['train']

    if train_opt.get('pixel_opt'):
        self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
        
    if train_opt.get('tdp_opt'):
        self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)
        
    if train_opt.get('ce_sr_opt'):
        self.cri_ce_sr = build_loss(train_opt['ce_sr_opt'], self.text_logger).to(self.device)
    
    if train_opt.get('ce_hr_opt'):
        self.cri_ce_hr = build_loss(train_opt['ce_hr_opt'], self.text_logger).to(self.device)
        
    if train_opt.get('ce_cqmix_opt'):
        self.cri_ce_cqmix = build_loss(train_opt['ce_cqmix_opt'], self.text_logger).to(self.device)

    self.setup_optimizers()
    self.setup_schedulers(len(data_loader_train), name='sr', optimizer=self.optimizer_sr)
    self.setup_schedulers(len(data_loader_train), name='cls', optimizer=self.optimizer_cls)
    
    os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
    os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
    
    self.eval_freq = train_opt.get('eval_freq', 1)
    self.warmup_epoch = train_opt.get('warmup_epoch', -1)
    self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))
    
 def setup_optimizers(self):
    train_opt = self.opt['train']
    
    optim_type = train_opt['optim_sr'].pop('type')
    self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
    self.optimizers.append(self.optimizer_sr)
    
    optim_type = train_opt['optim_cls'].pop('type')
    self.text_logger.write('NOTICE: Train all layers of net_cls')
    self.optimizer_cls = self.get_optimizer(optim_type, self.net_cls.parameters(), **train_opt['optim_cls'])
    self.optimizers.append(self.optimizer_cls)
    
 def train_one_epoch(self, data_loader_train, train_sampler, epoch):
    self.set_mode(mode='train')
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr_sr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("lr_cls", SmoothedValue(window_size=1, fmt="{value}"))
    
    if self.dist:
        train_sampler.set_epoch(epoch)
        
    if epoch < self.warmup_epoch + 1:
        self.text_logger.write("NOTICE: Doing warm-up")

    max_logged_acc = -1.0 
    header = f"Epoch: [{epoch}, Name {self.opt['name']}]"
    correct = 0
    total = 0
    for iter, (img_hr, label) in enumerate(metric_logger.log_every(data_loader_train, self.opt['print_freq'], self.text_logger, header)):
        img_hr, label = img_hr.to(self.device), label.to(self.device)
        current_iter = iter + len(data_loader_train)*(epoch-1)
        batch_size = img_hr.shape[0]

        img_lr = quantize(interpolate(img_hr, scale_factor=(1/self.scale), mode='bicubic'))
            
        # PHASE 1
        img_sr = self.net_sr(img_lr)
        for p in self.net_cls.parameters(): p.requires_grad = False
        self.optimizer_sr.zero_grad()
        l_total_sr = 0

        if hasattr(self, 'cri_pix'):
            l_pix = self.cri_pix(img_sr, img_hr)
            metric_logger.meters["l_pix"].update(l_pix.item()) 
            self.tb_logger.add_scalar('losses/l_pix', l_pix.item(), current_iter)
            l_total_sr += l_pix

        if epoch > self.warmup_epoch:
            if hasattr(self, 'cri_tdp'):
                self.net_cls.eval()
                _, feat_sr = self.net_cls(self.normalize(img_sr), return_feats=True)
                _, feat_hr = self.net_cls(self.normalize(img_hr), return_feats=True)
                self.net_cls.train()
                with torch.no_grad():
                    lpips_val = self.lpips_fn(img_sr, img_hr).mean()
                weight = 1 / (1 + lpips_val)
                l_tdp = self.cri_tdp(feat_sr, feat_hr)
                l_tdp = weight * l_tdp
                metric_logger.meters["l_tdp"].update(l_tdp.item()) 
                self.tb_logger.add_scalar('losses/l_tdp', l_tdp.item(), current_iter)
                l_total_sr += l_tdp

        l_total_sr.backward()
        self.optimizer_sr.step()
        
        # PHASE 2
        img_sr = self.net_sr(img_lr).detach()
        for p in self.net_cls.parameters(): p.requires_grad = True
        self.optimizer_cls.zero_grad()
        l_total_cls = 0

        if hasattr(self, 'cri_ce_sr'):
            pred_sr = self.net_cls(self.normalize(img_hr))
            l_ce_sr = self.cri_ce_sr(pred_sr, label)
            metric_logger.meters["l_ce_sr"].update(l_ce_sr.item())
            self.tb_logger.add_scalar('losses/l_ce_sr', l_ce_sr.item(), current_iter)
            l_total_cls += l_ce_sr

        if hasattr(self, 'cri_ce_hr'):
            pred_hr = self.net_cls(self.normalize(img_hr))
            l_ce_hr = self.cri_ce_hr(pred_hr, label)
            metric_logger.meters["l_ce_hr"].update(l_ce_hr.item())
            self.tb_logger.add_scalar('losses/l_ce_hr', l_ce_hr.item(), current_iter)
            l_total_cls += l_ce_hr

        if hasattr(self, 'cri_ce_cqmix'):
            h, w = img_sr.shape[-2:]
            mask = interpolate((torch.randn(batch_size,1,4,4)).bernoulli_(p=0.5),size=(h, w),mode='nearest').to(self.device)
            img_cqmix = img_sr*mask + img_hr*(1-mask)
            pred_cqmix = self.net_cls(self.normalize(img_cqmix))
            l_ce_cqmix = self.cri_ce_cqmix(pred_cqmix, label)
            metric_logger.meters["l_ce_cqmix"].update(l_ce_cqmix.item())
            self.tb_logger.add_scalar('losses/l_ce_cqmix', l_ce_cqmix.item(), current_iter)
            l_total_cls += l_ce_cqmix

        l_total_cls.backward()
        self.optimizer_cls.step()

        psnr, valid_batch_size = calculate_psnr_batch(quantize(img_sr), img_hr)
        metric_logger.meters["psnr"].update(psnr.item(), n=valid_batch_size)
        
        # Calculate accuracy for this specific batch
        acc1_sr_batch, _ = calculate_accuracy(pred_sr, label, topk=(1, 5))
  
        _, preds = pred_sr.max(1)
        correct += (preds == label).sum().item()
        total += label.size(0)
        metric_logger.meters["acc1_sr"].update(acc1_sr_batch.item(), n=batch_size)

     

        metric_logger.update(lr_sr=round(self.optimizer_sr.param_groups[0]["lr"], 8))
        metric_logger.update(lr_cls=round(self.optimizer_cls.param_groups[0]["lr"], 8))
        self.update_learning_rate()

 
    train_acc = 100.0 * correct / total if total > 0 else 0
    return train_acc

 def evaluate(self, data_loader, epoch):
    self.net_cls.eval()
    self.net_sr.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            img_hr, labels = data
            img_hr, labels = img_hr.to(self.device), labels.to(self.device)
            outputs = self.net_cls(img_hr)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    eval_acc = 100.0 * correct / total
    return eval_acc
