# Classifier tuning

## Baseline settings:
Batch size = 32
Input size = 48
Optimizer: Adam, lr=1e-3
Cross Entropy loss

## Architecture search
### ResNet18:
Best F1-score = 0.7716
Test F1-score = 0.7538

### ResNet34:
Best F1-score = 0.7841
Test F1-score = 0.7740

### ResNet50:
Best F1-score = 0.7793
Test F1-score = 0.7645

### ResNext50_32x4d
Best F1-score = 0.7960
Test F1 score = 0.7748

### ResNext101_32x8d
Best F1-score = 0.7770
Test F1 score = 0.7600

### Efficientnet_b3a
Best F1-score = 0.7878
Test F1 score = 0.7732

### Densenet121 baseline
Best F1-score = 0.7929
Test F1 score = 0.7695

### MobileNetv3 baseline
Best F1-score = 0.7637
Test F1 score = 0.7401

Select ResNet34

## Add base augs: Horizontal Flip + Rotate + RandomBrightnessContrast
Best F1-score = 0.8067
Test F1 score = 0.7978

## Add DropOut:
### p=0.5 (default)
Best F1-score = 0.8039
Test F1 score = 0.7936

### p=0.6
Best F1-score = 0.8106
Test F1 score = 0.7971

### p=0.7
Best F1-score = 0.8140
Test F1 score = 0.7951

### p=0.4
Best F1-score = 0.8091
Test F1 score = 0.7922

## New baseline:
ResNet34 + baseaugs (Horizontal Flip + Rotate + RandomBrightnessContrast)

## Tune input size:
### Size = 64
Best F1-score = 0.8151
Test F1 score = 0.7988

### Size = 128
Best F1-score = 0.8345
Test F1 score = 0.8128

### Size = 224 (New baseline)
Best F1-score = 0.8332
Test F1 score = 0.8318

### Size = 512
Best F1-score = 0.8361
Test F1 score = 0.8241

## Tune batch size:
### batch size = 64
Best F1-score = 0.8399
Test F1 score = 0.8283

### batch size = 16
Best F1-score = 0.8377
Test F1 score = 0.8309

## Tune optimizer:
### Try SGD with momentum baseline (bs=32, input size=224)
Best F1-score = 0.8393
Test F1 score = 0.8271

### Apply Cyclic scheduler
Best F1-score = 0.8449
Test F1 score = 0.83

## Tune weight decay:
### wd=1e-4
Best F1-score = 0.8423
Test F1 score = 0.8299

### wd=1e-5
Best F1-score = 0.8467
Test F1 score = 0.8326

## Add CutOut aug + CoarseDropOut
Best F1-score = 0.8412
Test f1 score = 0.8316

## Add weights to CE Loss:
Best F1-score = 0.8343
Test f1 score = 0.8178

# Best setting so far:
* Model: ResNet34
* Augs: Horizontal Flip + Rotate + RandomBrightnessContrast
* batch size = 32
* imput size = 224
* Optimizer and scheduler: SGD with momentum=0.9 and weight decay=1e-5
* Cyclic schedulr with triangular2 mode and min_lr=1e-2, max_lr=2e-2

Best F1-score = 0.8467
Test F1 score = 0.8326
