Using LoRA for Efficient Stable Diffusion Fine-Tuning
Published January 26, 2023
Pedro Cuenca's avatar
pcuenq
Pedro Cuenca
Sayak Paul's avatar
sayakpaul
Sayak Paul
LoRA: Low-Rank Adaptation of Large Language Models is a novel technique introduced by Microsoft researchers to deal with the problem of fine-tuning large-language models. Powerful models with billions of parameters, such as GPT-3, are prohibitively expensive to fine-tune in order to adapt them to particular tasks or domains. LoRA proposes to freeze pre-trained model weights and inject trainable layers (rank-decomposition matrices) in each transformer block. This greatly reduces the number of trainable parameters and GPU memory requirements since gradients don't need to be computed for most model weights. The researchers found that by focusing on the Transformer attention blocks of large-language models, fine-tuning quality with LoRA was on par with full model fine-tuning while being much faster and requiring less compute.

