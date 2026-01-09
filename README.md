# DDPM-CelebA: Face Generation with Diffusion Models

This repository contains a PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)**. The model is trained on the **CelebA** dataset to generate human face images by learning to reverse a 1,000-step Gaussian noise process.

## 🛠 Architecture & Features

* **Backbone**: Time-Conditional **U-Net** featuring **Residual Blocks** and **Group Normalization**.


* **Time Embeddings**: Sinusoidal position embeddings allow the model to predict noise specific to each timestep .


* **Fast Sampling**: Supports **DDIM (Denoising Diffusion Implicit Models)**, reducing inference time from 1,000 steps to **50 steps**.


* **Cloud Integration**: Built-in support for Google Colab and automated **Google Drive** checkpointing.



## 📊 Training Results (100 Epochs)

The model was trained on a subset of 5,000 images with the following performance metrics:

| Metric | Start (Epoch 1) | Final (Epoch 100) |
| --- | --- | --- |
| **AVG Train Loss** | 0.2240 | **0.0183** |
| **AVG Val Loss** | 0.0740 | **0.0186** |

### Key Observations:

* **Convergence**: Both training and validation losses converged to stable, low values, indicating successful learning and strong generalization.
* **Stability**: The model showed no signs of significant overfitting throughout the 100-epoch training marathon.
* **Pixel Distribution**: Generated images occasionally produced pixel values slightly outside the  range (e.g.,  to ), requiring automated clipping for optimal visualization in Matplotlib.

## ⚙️ Setup & Usage

1. **Dataset**: Provide your `kaggle.json` to download CelebA automatically.


2. **Training**: Run the main execution block to start the process:
```python
if __name__ == '__main__':
    train_ddpm()

```


3. **Outputs**: Sample images are plotted every 5 epochs, and checkpoints are saved to your configured Drive folder.

