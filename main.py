import datetime
import os.path as osp
import time
import torch
import torch.utils.data
import utils.common

from models import make_model
from data import load_data

def main():
    opt, args = utils.common.parse_options()
    opt = utils.common.init_distributed_mode(opt)
    
    if opt.get('deterministic', False):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        
    model = make_model(opt)
    utils.common.copy_opt_file(args.opt, osp.join('experiments', opt['task'], opt['name']))
    
    data_loader_train, data_loader_test, train_sampler, test_sampler = load_data(opt)
    
    if opt.get('train', False):
        model.init_training_settings(data_loader_train)
        if opt.get('resume', False):
            resume_epoch = model.resume_training(opt['resume'])
            start_epoch, end_epoch = resume_epoch+1, opt['train']['epoch']
        else:
            start_epoch, end_epoch = 1, opt['train']['epoch']
        
        model.text_logger.write("Start training")
        start_time = time.time()
        best_acc = 0
        
        for epoch in range(start_epoch, end_epoch+1):
          train_acc = model.train_one_epoch(data_loader_train, train_sampler, epoch)
          torch.save(model.net_cls.state_dict(), f"/content/net_cls_epoch_{epoch}.pth")

          if train_acc is not None and train_acc > best_acc:
            best_acc = train_acc
          

          # This will now correctly show the best of the printed batch values
          print(f"Epoch {epoch} Train Acc: {train_acc:.2f}% | Best: {best_acc:.2f}%")
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        model.text_logger.write(f"Training time {total_time_str}")

    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
        if opt.get('calculate_cost'):
            model.calculate_cost()
        else:
            model.evaluate(data_loader_test, test_sampler)

if __name__ == "__main__":
    main()
