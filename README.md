# Deepsight

## Amazon ML Challenge 2025 – Solution

### Key Highlights

* **Multi-modal Transfer Learning**
  Utilized transfer learning to generate 768-dimensional embeddings from paired `text` and `image` inputs. Trained a price regression model on a dataset of **75,000** product entries, each containing an image, text description, and price.

* **V1 Architecture: CLIP + Bi-Directional Cross-Attention**

  * Employed CLIP with all layers frozen except the final layers of the image and text encoders.
  * Text and image embeddings were passed through a **bi-directional cross-attention** mechanism:

    * Text tokens attend to image tokens and vice versa.
    * Attention scores are added to the original embeddings.
  * Outputs are **layer-normalized**, concatenated, and normalized again.
  * Final regression head is a **Transformer MLP with skip connections**.
  * Learning rates:

    * `1e-7` for frozen CLIP encoder fine-tuning.
    * `1e-5` for the custom regression head.
  * Prices are standardized before training, and predictions are **de-normalized** post-inference.

* **V2 Architecture: SigLIP + EmbeddingGemma + Advanced Training Strategies**

  * Switched to **SigLIP (200M)** and **EmbeddingGemma (300M)** for pretrained embeddings.
  * Maintained the bi-directional cross-attention mechanism.
  * Enhanced training with:

    * **K-Fold Cross-Validation**
    * **CosineAnnealingWarmRestarts** learning rate scheduler
    * **Huber loss** optimized on `log1p` of prices with predictions post-processed via `expm1p`.
  * Optimization focused on minimizing **SMAPE (Symmetric Mean Absolute Percentage Error)** for better real-world performance.
