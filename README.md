# Deepsight

## Amazon ML challange 2025 solution.

### Key points

* Leveraged transfer learning for 768 dimensional embeddings of `text` and `image` pairs for product price regression on a 75k image+text+price dataset.
* V1: Freezed layers of clip other than final layers of the image and text encoder with very small lr of 1e-7, and created a bi-dirctional cross-attention architechture for regression with lr of 1e-5 where text tokens are tending to images and image tokens are tending to text. And these scores are added with the outputs of the encoders. These are then layerwise normalized, concatenated togather and then further normalized. Finally transformer mlps are used with skip connections, for prediction.
* V2: Leveraged Siglip(200m) and EmbeddingGemma(300m) for pretrained embeddings on both train and test sets. Used similar bi-directional cross attention architechture along with: K-Fold cross validation and `CosineAnnealingWarmRestarts` scheduling for further enhancing the predictor's performance on **SMAPE** criterion.
