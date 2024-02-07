---
title: Generative AI- Stable Diffusion
date: 2023-02-07 23:10:08
categories:
- Language
tags: [Graph, Laplacian]
typora-root-url: ../../allenlu2009.github.io


---



## Main Reference

https://arxiv.org/pdf/1503.03585.pdf : original Stanford Diffusion paper: very good!

https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ : good blog article including conditional diffusion

https://jalammar.github.io/illustrated-stable-diffusion/  by Jay Alammar, excellent and no math!

[@alammarIllustratedStable2022a] by Jay Alammar, excellent and no math!

[@alammarIllustratedStable2022] by Jay Alammar, excellent and no math!



## Introduction

Stable Diffusion 的歷史

2015: Sohl-Dickstein, et al (Stanford) "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" [@hoDenoisingDiffusion2020]:  首次提出使用 

* Forward: Markov diffusion kernel (Gaussian or Binomial diffusion)  
* Backward: ? what deep learning model?
* Entropy
* 

Deep Unsupervised Learning using Nonequilibrium: Entropy

2020 DDPM (Denoising Diffusion Probabilistic Model):   probabilistic model Markov/VAE

2021

Classifier-free diffusion Guidance

GLIDE

DALLE, Googel Imagen, Midjourney







## Stable Diffusion



Stable Diffusion 和 diffusion model 的差異之處

1. 發生在 **compressed** latent spaces, 也就是 z 的 dimension 小於 x dimension.  這和一般 diffusion model 假設 x and z 是同樣 dimension 不同。不過好像也 make sense?
2. Transformer 的 hint (or token) 用在 latent space 可以 condition the generation.
3. Stable diffusion 是 noise prediction instead of denoise.





<img src="/media/image-20230208221847859.png" alt="image-20230208221847859" style="zoom:80%;" />



<img src="/media/image-20230212001348131.png" alt="image-20230212001348131" style="zoom:80%;" />



### 重點不是數學，而是 Visualization

(Reference)

Stable Diffusion 功能：text-to-image,   text+image-to-image,   (? image-to-image?)



<img src="/media/image-20230211160145083.png" alt="image-20230211160145083" style="zoom: 50%;" />



### Stable Diffusion 組成元件

Stable Diffusion 是一個系統包含多個組件和模型，而非單一的模型。可以簡單分成兩大塊：文本理解組件和圖像生成組件。

* 文本理解 (text encoder):  transformer network 的 encoder 部分
  * 類似 BERT?  Not exactly。因爲要同時處理 text and image, 需要 multi-modal vision and language model，比較像是 CLIP (Contrastive **Language-Image** Pre-Training).  Transformer 家族的 BERT, ViT, CLIP, 和 BLIP 的差異可以參考 appenix.
* 圖像生成 (image generator):  diffusion network based on U-Net?



<img src="/media/image-20230211201429596.png" alt="image-20230211201429596" style="zoom:50%;" />



#### 文本理解器 (Text Understander)

將文本信息翻譯成數字表示 (numeric representation)，以捕捉文本中的語義信息。

雖然目前還是從宏觀角度分析模型，之後才有更多的模型細節。但我們可以大致推測這個文本編碼器 (text encoder) 是一個特殊的Transformer model.   具體來說是 CLIP -  Contrastive Language-Image Pre-Training 模型的文本編碼器。

我們先考慮文本輸入：模型的輸入為一個文本字元串，輸出為一個數字列表，用來表徵文本中的每個單詞/token，即將每個 token 轉換為一個向量。
然後這些信息會被提交到圖象生成器

#### 圖像生成器 (Image Generator)

圖象生成器主要包括兩個階段：

<img src="/media/image-20230211222238654.png" alt="image-20230211222238654" style="zoom:50%;" />

##### 1. Image information creator (latent space)

這個組件是 Stable Diffusion 的獨家秘方。相比之前的模型，它的很多性能改善都是在這個組件實現。
該組件運行多個 denoise steps 來生成圖像信息，其中 steps 也是Stable Diffusion 介面和 library 中的參數，通常預設為 50 或 100。

**Image information creator 完全在 image information space（latent space）中運行，這一特性使得它比其他在 pixel space 工作的 Diffusion 模型運行得更快。從技術上來看，該組件由一個 UNet 和一個scheduling 算法組成。** 

* Scheduling algorithm 是產生高畫質的關鍵！
* UNet 目前面臨 transformer decoder 的挑戰。

Diffusion 描述了在該組件內部運行期間發生的事情，即對 image information (不是 image 本身) 進行一步步地處理，並最終由下一個組件（圖像解碼器）生成高質量的圖象。

##### 2. 圖像解碼器 (Image Decoder)

圖像解碼器根據從 Image Information Creator 中獲取的信息產生出一幅畫。**整個過程只運行一次即可生成最終的像素級圖像。這個 decoder 類似 VAE 的 decoder.**



### 神經網絡模型

可以看到，Stable Diffusion總共包含三個主要的組件，其中每個組件都擁有一個獨立的神經網絡：

* Clip Text 用於文本 (和圖像) 編碼。
  輸入：文本
  輸出：77 個 token embeddings 向量，其中每個向量包含 768 個維度
* UNet + Scheduler 在 information (latent) space 中逐步處理/擴散信息。
  輸入：embeddings 和一個由雜訊初始 tensor.
  輸出：一個經過處理的 information tensor.
* 自編碼解碼器（Autoencoder Decoder），使用處理過的 information tensor 繪製最終圖像的解碼器。
  輸入：處理過的 information tensor，維度為（4, 64, 64）
  輸出：最終圖像，維度為（3，512，512）即（RGB，寬，高）



## Diffusion

最關鍵的部分就是 diffusion.   Diffusion 是在下圖中粉紅色的 Image Information Creator 組件中發生的過程，過程中包含表徵輸入文本的 (1) token embeddings，和隨機初始的 (2) image tensor（也稱之為 latents），該過程會還需要用到圖象解碼器來繪製最終 (3) 圖象的信息矩陣。

<img src="/media/image-20230211230509449.png" alt="image-20230211230509449" style="zoom: 67%;" />



整個運行過程是 step-by-step，每一步都會增加更多的相關信息。
為了更直觀地感受整個過程，可以中途查看隨機 latents 矩陣，並觀察它是如何轉化為視覺雜訊的，其中視覺檢查（visual inspection）是通過圖象解碼器進行的。

整個diffusion過程包含多個steps，其中每個step都是基于輸入的 latents 矩陣進行操作，並生成另一個 latents 矩陣以更貼近「輸入的文本」和從模型圖象集中獲取的「視覺信息」。

將這些 latents 可視化可以看到這些信息是如何在每個 step 中相加的。

整個過程就是從無到有，看起來相當激動人心。

<img src="/media/image-20230213222444971.png" alt="image-20230213222444971" style="zoom:50%;" />

步驟2和4之間的過程轉變看起來特別有趣，就好像圖片的輪廓是從雜訊中出現的。

### Diffusion 的工作原理

有兩種觀點：image denoise 或是 noise prediction.   目前是以 noise prediction 爲主, why?

* Predict noise 之後的 loss 計算和 back-propagation 都和原始 image 無關，似乎比較簡單?
* 如果是 image denoise, 結果需要和原始 image 比較 (相減) 的 loss 才能 back-propagation, 比較麻煩?
* 所以這和一般影像的 denoise 或是 predict noise 似乎不同？還是要用影像的 noise 來 train, 而非用 Gaussian noise?



使用擴散模型生成圖象的核心思路還是基于已存在的強大的計算機視覺模型，只要輸入足夠大的數據集，這些模型可以學習任意複雜的操作。

#### Forward path: 

假設我們已經有了一張圖象，生成產生一些雜訊加入到圖象中，然後就可以將該圖象視作一個訓練樣例。

<img src="/media/image-20230213222840925.png" alt="image-20230213222840925" style="zoom: 67%;" />

使用相同的操作可以生成大量訓練樣本來訓練圖象生成模型中的核心組件。

**這個部分和我的瞭解不同！應該是逐漸加 noise 而非加上不同程度的 noise。還是兩者等價?**

<img src="/media/image-20230213222915909.png" alt="image-20230213222915909" style="zoom: 67%;" />

上述例子展示了一些可選的雜訊量值，從原始圖象(級別0，不含雜訊)到雜訊全部添加(級別4) ，從而可以很容易地控制有多少雜訊添加到圖象中。
所以我們可以將這個過程分散在幾十個steps中，對數據集的每張圖象都可以生成數十個訓練樣本。

<img src="/media/image-20230214210820883.png" alt="image-20230214210820883" style="zoom: 80%;" />

基于上述數據集，我們就可以訓練出一個性能極佳的**雜訊預測器**，每個訓練 step 和其他模型的訓練相似。當以某一種確定的配置運行時，雜訊預測器就可以生成圖象。

<img src="/media/image-20230214210933913.png" alt="image-20230214210933913" style="zoom:80%;" />

### 移除雜訊，繪製圖象

經過訓練的雜訊預測器可以對一幅添加雜訊的圖象進行去噪，也可以預測添加的雜訊量。

<img src="/media/image-20230214225820874.png" alt="image-20230214225820874" style="zoom:67%;" />

由於採樣的雜訊是可預測的，所以如果從圖象中減去雜訊，最後得到的圖象就會更接近模型訓練得到的圖象。

得到的圖象並非是一張精確的原始圖象，而是分佈（distribution），即世界的像素排列，比如天空通常是藍色的，人有兩隻眼睛，貓有尖耳朵等等，生成的具體圖象風格完全取決於訓練數據集。

<img src="/media/image-20230214230025753.png" alt="image-20230214230025753" style="zoom:80%;" />

以上的步驟成爲 DDPM (Denoising Diffusion Probabilistic Models).  不止Stable Diffusion通過去噪進行圖象生成，DALL-E 2和谷歌的Imagen模型都是如此。

需要注意的是，到目前為止描述的擴散過程還沒有使用任何文本數據生成圖象。因此，如果我們部署這個模型的話，它能夠生成很好看的圖象，但用戶沒有辦法控制生成的內容。
在接下來的部分中，將會對如何將條件文本合併到流程中進行描述，以便控制模型生成的圖象類型。



#### **加速：在压缩 (Latent) 数据上扩散**

為了加速圖象生成的過程，Stable Diffusion並沒有選擇在像素圖象本身上運行擴散過程，而是選擇在圖象的壓縮版本上運行，論文中也稱之為「Departure to Latent Space」。
整個壓縮過程，包括後續的解壓、繪製圖象都是通過自編碼器 (auto-encoder) 完成的，將圖象壓縮到潛空間中，然後僅使用解碼器使用壓縮後的信息來重構。

<img src="/media/image-20230215231532931.png" alt="image-20230215231532931" style="zoom:100%;" />

前向擴散（forward diffusion）過程是在壓縮latents完成的，雜訊的切片（slices）是應用於latents上的雜訊，而非像素圖象，所以雜訊預測器實際上是被訓練用來預測壓縮表示（潛空間）中的雜訊。



<img src="/media/image-20230215231653267.png" alt="image-20230215231653267" style="zoom:100%;" />

前向過程，即使用使用自編碼器中的編碼器來訓練雜訊預測器。一旦訓練完成後，就可以通過運行反向過程（自編碼器中的解碼器）來生成圖象。

<img src="/media/image-20230215231956901.png" alt="image-20230215231956901" style="zoom:80%;" />



前向和後向過程如下所示，圖中還包括了一個conditioning組件，用來描述模型應該生成圖象的文本提示。



<img src="/media/image-20230208221847859.png" alt="image-20230208221847859" style="zoom:80%;" />



<img src="/media/image-20230212001348131.png" alt="image-20230212001348131" style="zoom:80%;" />

### 文本编码器：一个Transformer语言模型



模型中的语言理解组件使用的是Transformer语言模型，可以将输入的文本提示转换为token嵌入向量。发布的Stable Diffusion模型使用 ClipText (基于 GPT 的模型) ，这篇文章中为了方便讲解选择使用 BERT模型。



![img](https://pic.rmb.bdstatic.com/bjh/down/917dc069c6479646818b97380591bbe5.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



Imagen论文中的实验表明，相比选择更大的图像生成组件，更大的语言模型可以带来更多的图像质量提升。



早期的Stable Diffusion模型使用的是OpenAI发布的经过预训练的 ClipText 模型，而在Stable Diffusion V2中已经转向了最新发布的、更大的CLIP模型变体OpenClip.



**CLIP是怎么训练的？**



CLIP需要的数据为图像及其标题，数据集中大约包含4亿张图像及描述。



![img](https://pic.rmb.bdstatic.com/bjh/down/49cc3cfba6307edf8f8d5a7980d12c78.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



数据集通过从网上抓取的图片以及相应的「alt」标签文本来收集的。



CLIP 是图像编码器和文本编码器的组合，其训练过程可以简化为拍摄图像和文字说明，使用两个编码器对数据分别进行编码。



![img](https://pic.rmb.bdstatic.com/bjh/down/35743f22eed9e61d1e2b25342f069b2f.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



然后使用余弦距离比较结果嵌入，刚开始训练时，即使文本描述与图像是相匹配的，它们之间的相似性肯定也是很低的。



![img](https://pic.rmb.bdstatic.com/bjh/down/8bb76c5390c6c857806564f82958777f.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



随着模型的不断更新，在后续阶段，编码器对图像和文本编码得到的嵌入会逐渐相似。



![img](https://pic.rmb.bdstatic.com/bjh/down/abb78633d59022792d2e4e5d37427de4.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



通过在整个数据集中重复该过程，并使用大batch size的编码器，最终能够生成一个嵌入向量，其中狗的图像和句子「一条狗的图片」之间是相似的。



就像在 word2vec 中一样，训练过程也需要包括不匹配的图片和说明的负样本，模型需要给它们分配较低的相似度分数。



### 文本信息喂入图像生成过程



为了将文本条件融入成为图像生成过程的一部分，必须调整噪声预测器的输入为文本。



![img](https://pic.rmb.bdstatic.com/bjh/down/7129bb646baa97b650632e2f4f6d1be3.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



所有的操作都是在潜空间上，包括编码后的文本、输入图像和预测噪声。



![img](https://pic.rmb.bdstatic.com/bjh/down/71d4769ff64b62b4e8c98b7864f408a2.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



为了更好地了解文本token在 Unet 中的使用方式，还需要先了解一下 Unet模型。



**Unet 噪声预测器中的层（无文本）**



一个不使用文本的diffusion Unet，其输入输出如下所示：



![img](https://pic.rmb.bdstatic.com/bjh/down/4bf0824ec6b60ae6bfb96c9b15d74625.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



在模型内部，可以看到：



1. Unet模型中的层主要用于转换latents；
2. 每层都是在之前层的输出上进行操作；
3. 某些输出（通过残差连接）将其馈送到网络后面的处理中
4. 将时间步转换为时间步长嵌入向量，可以在层中使用。



![img](https://pic.rmb.bdstatic.com/bjh/down/4d36623e10375c2f78aca60c0ff90878.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



**Unet 噪声预测器中的层（带文本）**



现在就需要将之前的系统改装成带文本版本的。



![img](https://pic.rmb.bdstatic.com/bjh/down/7d9e323f00d10df9783f3f24e66880cf.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



主要的修改部分就是增加对文本输入（术语：text conditioning）的支持，即在ResNet块之间添加一个注意力层。



![img](https://pic.rmb.bdstatic.com/bjh/down/c481e0208afa7cf998ff37369687cdeb.png?x-bce-process=image/watermark,bucket_baidu-rmb-video-cover-1,image_YmpoL25ld3MvNjUzZjZkMjRlMDJiNjdjZWU1NzEzODg0MDNhYTQ0YzQucG5n,type_RlpMYW5UaW5nSGVpU01HQg==,w_28,text_QOaWsOaZuuWFgw==,size_28,x_22,y_22,interval_2,color_FFFFFF,effect_softoutline,shc_000000,blr_2,align_1)



需要注意的是，ResNet块没有直接看到文本内容，而是通过注意力层将文本在latents中的表征合并起来，然后下一个ResNet就可以在这一过程中利用上文本信息。

### 



## Appendix

### Transformer Family Encoder 比較：ViT, CLIP, BLIP, BERT 

Ref: [@jizhiTransformerFamily2022]

完整的 transformer 包含 encoder (discriminative) 和 decoder (generative).

以下我們主要聚焦於 encoder 部分。因爲 stable diffusion 的 decoder (generative) 部分是由 U-Net 完成的。雖然目前也有 transformer-based 的 decoder.  

<img src="/media/image-20230211213507007.png" alt="image-20230211213507007" style="zoom:33%;" />



|                               | Input        | Output            |
| ----------------------------- | ------------ | ----------------- |
| Transformer (encoder+decoder) | Text         | Text              |
| BERT (encoder)                | Text         | Token Embeddings  |
| ViT (encoder)                 | Image        | Token Embeddings  |
| CLIP (encoder)                | Text & Image | Similarity Score  |
| BLIP (encoder+decoder)        | Text & Image | Token Embeddings? |



#### BERT: Bidirectional Encoder Representations from Transformer

<img src="/media/image-20230211215131978.png" alt="image-20230211215131978" style="zoom:50%;" />

#### ViT:  Vision Transformer Encoder

<img src="/media/image-20230211213334026.png" alt="image-20230211213334026" style="zoom:50%;" />



#### CLIP: Contrastive **Language-Image** Pre-Training Encoder

CLIP是一個 **multi-modal vison and language model**。它可用於 (image-text) **圖像和文本的相似性**以及 zero-**shot 圖像分類 (見下圖)**。CLIP 使用 ViT-like transformer 獲取**視覺特徵**，並使用因果語言模型 (causal language model) 獲取**文本特徵**。然後將**文本和視覺特徵投影到具有相同維度的 latent space**。最後投影圖像和文本特徵之間的内積產生相似性的分數 (score of similarity)。

<img src="/media/image-20230211213942601.png" alt="image-20230211213942601" style="zoom: 50%;" />



#### BLIP: Bootstrapping Language-Image Pre-training Encoder/Decoder

<img src="/media/image-20230211214454365.png" alt="image-20230211214454365" style="zoom:50%;" />

<img src="/media/image-20230211214821598.png" alt="image-20230211214821598" style="zoom:50%;" />







## Citation



