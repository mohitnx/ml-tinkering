# Alzheimer’s & Fashion Multimodal — Notebooks

Two PyTorch notebooks implementing classification experiments:
1. **Alzheimer’s detection (MRI)** — CNN-based (EfficientNet/CNN pipeline) MRI classification for Alzheimer’s stages.  
2. **Multimodal Fashion (image + text)** — Image + text fusion using cross-modal multi-head attention and fusion layers for fashion category classification.

---

## Notebooks (included)
- `alzheimers.ipynb`  
  *Task:* MRI image classification (Alzheimer’s / dementia stages) using a PyTorch dataset class, data augmentations, CNN/EfficientNet backbone, training loop, and standard metrics (accuracy, confusion matrix, etc.).  
  *Original notebook / results on Kaggle:* https://www.kaggle.com/code/mohitneupane/alzheimers

- `fashion-with-better-attention-mechanism.ipynb`  
  *Task:* Multimodal fashion classification (image + text). Uses cross-modal multi-head attention, text + image embeddings, focal loss options and fusion module. Includes training/validation loops and checkpointing (e.g., `fusion_best_v2.pt`).  
  *Original notebook / results on Kaggle:* https://www.kaggle.com/code/mohitnxn/fashion-with-better-attention-mechanism

---

## Quick summary of what’s inside
- Data handling: custom `Dataset` classes supporting common image folder structures / CSV metadata.
- Models:
  - Alzheimer’s: PyTorch `nn.Module` CNN / EfficientNet style backbone + classifier head.
  - Fashion: text encoder + image encoder + cross-modal MultiHeadAttention fusion + classifier head.
- Training: standard training loops, validation, checkpoint save/load, and metric logging (accuracy, loss, F1/confusion matrix where applicable).
- Utilities: data transforms, seeding, dataset loaders, and checkpoint utilities.

---

## Requirements
A minimal environment (adjust versions per your runtime):
- Python 3.8+
- torch
- torchvision
- numpy
- pandas
- scikit-learn
- Pillow
- tqdm
- (optional) transformers — if any notebook uses pretrained text encoders
- (optional) CUDA and correct nvidia/cuda drivers for GPU training

Install with:
```bash
pip install torch torchvision numpy pandas scikit-learn pillow tqdm
# add `transformers` if needed:
pip install transformers
