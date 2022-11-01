# IRHW

Repository for the paper "IRHW: A Strongly Covert High-Frequency Watermark for Model Copyright Protection".

## Code framework

- `irn`: The training and testing code of IRN model, which is used to generate watermarked dataset.
- `model`: The training and testing code of model for image classifying, the models included are: 
  - origin model:  ResNet-50 model trained using the original dataset
  - key model:  ResNet-50 model trained using the watermarked dataset
  - model stealing technology applied to key model, all settings are: fine-tuning (all layers or last layer), pruning (different pruning rates of 0.1, 0.3, 0.5), model distilling (from  ResNet-50 to GoogLeNet)
  - irrelevant model: VGG16 model trained using the original dataset
- `verify`: The training and testing code of model for feature vectors classifying (2 classes).
  - `binary_classifier`: code for generating feature vectors
  - `feature_extractor`: the training and testing code of binary classifer
