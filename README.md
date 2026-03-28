## SR4IR with Adaptive Weighting (LPIPS-based)

This project is a modified implementation of the SR4IR (Super-Resolution for Image Recognition) framework, adapted to run efficiently on limited GPU resources such as Google Colab (T4 GPU) and small custom datasets.

The original repository was designed for large-scale datasets like Stanford Cars with heavy models and high memory requirements. In this version, the model and training pipeline have been simplified and modified to work on a smaller dataset while also introducing LPIPS-based adaptive weighting during training.


## Overview

The goal of this project is to study how super-resolution affects image classification and to improve training by giving more importance to high-quality SR images. Instead of treating all SR outputs equally, the model uses LPIPS to estimate perceptual quality and assigns higher weights to better SR outputs.

The project also includes fixes for training instability, incorrect accuracy computation, and model collapse issues.


## Key Modifications

Several important changes were made compared to the original repository.

The SR network was simplified by reducing the number of blocks from 16 to 4 and the number of feature channels from 64 to 32. This significantly reduces GPU memory usage and allows training on Colab.

The classification network was adapted for a custom dataset with 40 classes instead of 196. Pretrained ImageNet weights are still used for ResNet18, but the final fully connected layer is adjusted automatically.

The dataset pipeline was changed to use a custom folder-based dataset instead of the original Stanford Cars structure.

Training resolution was reduced from 224/256 to 128 to further reduce computation.

The learning rate was reduced to stabilize training. The classifier learning rate was changed from 0.03 to 0.005, and the SR network learning rate was reduced to 5e-5.

A MultiStepLR scheduler was used instead of cosine annealing for simpler and more stable training on small datasets.

LPIPS-based adaptive weighting was introduced in the SR training phase. The perceptual difference between SR and HR images is computed and used to scale the feature loss. Better SR images receive higher importance.

The classification training was modified to use HR images directly instead of SR images to prevent model collapse. This stabilizes learning and improves accuracy.

The accuracy calculation was corrected. The original code tracked only batch-wise peak accuracy, which was misleading. This was replaced with proper epoch-level accuracy computed using total correct predictions over total samples.

Model saving in the original repository was unreliable in this setup, so manual saving using torch.save was added.


## Dataset Structure

The dataset should be organized as follows:

car_dataset/
train/
class1/
class2/
...
valid/
class1/
class2/
...
test/
class1/
class2/
...

Each class folder should contain images belonging to that class.

A typical setup used in this project is around 30 images per class for training and 5 images each for validation and testing.


## Configuration

The configuration file used is 114_SR4IR_edsr_x4.yml.

Important changes in this file include reducing model size, lowering batch size to 8, reducing image resolution to 128, and adjusting learning rates and schedulers.

The dataset path must be updated to your local or Google Drive path.


## How to Run (Google Colab)

Clone the repository:

git clone <your_repo_link>
cd SR4IR/src

Mount Google Drive if using Colab:

from google.colab import drive
drive.mount('/content/drive')

Make sure your dataset path in the YAML file points to your dataset directory.

Run training:

python main.py -opt ../options/cls/StanfordCars/114_SR4IR_edsr_x4.yml

Training will run for the specified number of epochs and print training accuracy after each epoch.

Models are saved manually during training using torch.save.


## Testing

Testing can be done using the saved classifier model.

Load the model:

model.net_cls.load_state_dict(torch.load("/content/net_cls_epoch_X.pth"))

Run inference on the test dataset using a standard PyTorch DataLoader.

Accuracy is computed as total correct predictions divided by total samples.


## Evaluation Metrics

The project includes evaluation using confusion matrix, classification report, and visualization of predictions.

Confusion matrix helps identify which classes are being confused.


## Notes

This implementation is designed for small-scale experimentation and may not match the performance of the original SR4IR paper on large datasets.

Using more images per class significantly improves performance. Increasing dataset size is the most effective way to improve accuracy.

If the model predicts only one class, it indicates training instability. This was fixed by correcting accuracy computation and stabilizing training.


## Files Included

Both .py and .ipynb versions are provided. The notebook versions are included because GitHub sometimes does not render notebooks correctly.

The main modifications are in sr4ir_cls_model.py, main.py, options.py and the YAML configuration file.

Replication is shown in replication.ipynb/ replication.py

Novelty of adding adaptive weights for SR4IR architecture is shown in SR4IR_adaptiveweights.ipynb/ sr4ir_adaptiveweights.py

