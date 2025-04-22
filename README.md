# imagenet_task_b

Design a compact CNN with 2–4 effective layers for image classification on the ImageNet-mini dataset.

## Two-Layer Network for Image Classification

This repository contains the implementation of three image classification models:
- A baseline ResNet34 trained from scratch
- A simplified CNN model (SimpleNetV2)
- A proposed lightweight variant (SimpleNetV2_LiteX) with improved accuracy and reduced parameters

## Folder Structure

- `models/`: includes SimpleNetV2 and SimpleNetV2_LiteX architecture definitions
- `dataset/`: custom dataset loader for ImageNet-mini
- `data/`: contains `train.txt`, `val.txt`, and `test.txt` for data splits
- `logs/`: training logs and model checkpoints
- `train_resnet34.py`: training script for ResNet34
- `train_taskb_v2.py`: training script for SimpleNetV2
- `train_taskb_v2_litex.py`: training script for SimpleNetV2_LiteX
- `test_taskb_v2_litex.py`: inference script for LiteX
- `test_result_*.csv`: final test results for all models
- `requirements.txt`: Python dependency list
- `report/`: final report (TaskB_report.md/pdf)

## How to Run

```bash
# Train ResNet34 baseline
python train_resnet34.py

# Train SimpleNetV2
python train_taskb_v2.py

# Train SimpleNetV2_LiteX (proposed model)
python train_taskb_v2_litex.py

# Run inference on LiteX model
python test_taskb_v2_litex.py --checkpoint logs/taskb_v2_litex_epoch20.pth
