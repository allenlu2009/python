---
title: Mamba Vs. Transformer
date: 2024-01-28 23:10:08
categories:
- Language
tags: [GPT, LLM, HuggingFace, prompt]
description: LLM Tokenizer
typora-root-url: ../../allenlu2009.github.io




---






## Takeaway

S4 -> S6 -> Mamba



|                   | RNN/LSTM           | Transformer         | RWKV              | Mamba          |
| ----------------- | ------------------ | ------------------- | ----------------- | -------------- |
| Train, 時間維度   | 梯度消失，無法平行 | 可以平行            | 可以平行          | 可以平行       |
| Attention scope   | 小，附近 tokens    | 大，$T$             | 大，$T$           | 無窮大?        |
| Attention 計算    | 綫性               | 平方 (prefill)      | 綫性              | 綫性 or fixed? |
| Attention 存儲    | 綫性               | 平方 (update)       | 綫性              | 綫性 or fixed? |
| Complexity, Time  |                    | $O(T^2 d)$          | $O(T d)$          |                |
| Complexity, Space |                    | $O(T^2 + Td)$       | $O(d)$            |                |
| Nonlinearity      | Sigmoid?           | Softmax, layer norm | Softmax, sigmoid? | selector       |

$T$: sequence length;  $d$: feature dimension.





## 介紹

自 2017 年被提出以来，Transformer 已经成为 AI 大模型的主流架构，但随着模型规模的扩展和需要处理的序列不断变长，Transformer 的局限性也逐渐凸显。一个很明显的缺陷是：Transformer 模型中自注意力机制的计算量会随着上下文长度的增加呈平方级增长，比如上下文增加 32 倍时，计算量可能会增长 1000 倍，计算效率非常低。

为了克服这些缺陷，研究者们开发出了很多注意力机制的高效变体，但这往往以牺牲其有效性特为代价。到目前为止，这些变体都还没有被证明能在不同领域发挥有效作用。幾個例子: RetNet, RWKV (Receptance Weighted Key Value).







### RNN/LSTM

* Training 長 sequence 的困難:  (1) 梯度消失; (2)  recurrent 結構所以無法在時間維度平行訓練。但在 batch 方向仍然可以平行。 
* Inference (batch=1):  雖然是 recurrent generation 無法像 CNN 可以平行展開。所有生成的 token 都 depend on 之前 tokens.  **但好處只需要前一個 time step 的 hidden information + input (Markovian)!**    **因爲訓練長 sequence 困難，另一個缺點是 attention scope 不夠！比較久之前的 tokens attention 會消失**，因此常用於語音。



### Transformer

* Training:  可以在時間維度 (token sequence) 平行訓練 (類似 prompt mode).  這是最大的優點。
* Inference (batch=1, generative mode):  (1) 好處是 attention scope 包含所有之前 context 範圍的 tokens (1K/4K/8K); (2) 缺點是 attention matrix 的計算和存儲都和 context length 的平方成正比。另一個缺點是 token generation 仍然是 recurrent.   

Transformer 彻底改变了几乎所有[自然语言处理](https://www.zhihu.com/search?q=自然语言处理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3211669817}) (NLP) 任务，但其内存和计算复杂性却与序列长度呈二次方关系。相比之下，RNN 和 LSTM 在内存和计算要求方面表现出[线性扩展](https://www.zhihu.com/search?q=线性扩展&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3211669817})，但由于并行化和可扩展性的限制，很难达到与 Transformer 相同的性能。



### RWKV

一种新颖的模型架构，即接收加权键值（Receptance Weighted Key Value, **RWKV**），它将 Transformer 的高效可并行训练与 RNN 的高效推理相结合。我们的方法利用**线性注意力**机制，允许我们将模型制定为 Transformer 或 RNN，它在训练过程中并行计算，并在推理过程中保持恒定的计算和内存复杂性，从而使第一个非 Transformer 架构扩展到数十个数十亿个参数。**我们的实验表明，RWKV 的性能与类似大小的 Transformer 相当，这表明未来的工作可以利用这种架构来创建更高效的模型。**这项工作在协调序列处理任务中计算效率和模型性能之间的权衡方面迈出了重要一步。

缺點:  雖然是綫性。但是非綫性計算非常複雜？看起來是用計算換綫性 attention?





### Mamba (S4 -> S6)

最近，一项名为「Mamba」的研究似乎打破了这一局面。这篇论文的作者只有两位，一位是卡内基梅隆大学机器学习系助理教授 Albert Gu，另一位是 Together.AI 首席科学家、普林斯顿大学计算机科学助理教授（即将上任）Tri Dao。

一个重要创新是引入了一个名为「选择性 SSM」的架构，该架构是 Albert Gu 此前主导研发的 S4 架构（Structured State Spaces for Sequence Modeling ，用于序列建模的结构化状态空间）的一个简单泛化，可以有选择地决定关注还是忽略传入的输入。一个「小小的改变」—— 让某些参数成为输入的函数，结果却非常有效。



值得一提的是，S4 是一个非常成功的架构。此前，它成功地对 Long Range Arena (LRA) 中的长程依赖进行了建模，并成为首个在 Path-X 上获得高于平均性能的模型。更具体地说，S4 是一类用于深度学习的序列模型，与 RNN、CNN 和经典的状态空间模型（State Space Model，SSM）广泛相关。SSM 是独立的序列转换，可被整合到端到端神经网络架构中（ SSM 架构有时也称 SSNN，它与 SSM 层的关系就像 CNN 与线性卷积层的关系一样）。Mamba 论文也讨论了一些著名的 SSM 架构，比如 Linear attention、H3、Hyena、RetNet、RWKV，其中许多也将作为论文研究的基线。Mamba 的成功让 Albert Gu 对 SSM 的未来充满了信心。

Tri Dao 则是 [FlashAttention](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650848429%26idx%3D4%26sn%3D4665869919c379023b1bdb29568cdb2c%26chksm%3D84e578d3b392f1c59b3e4c9b986a522f6534e894800e12f5df52964d1ec45f6c067a73b53bd9%26scene%3D21%23wechat_redirect)、[Flash Attention v2](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650884509%26idx%3D3%26sn%3D65476dbc71ca235155734ed6cf52197d%26chksm%3D84e48de3b39304f54bc222ce6da480ef5ddd8874b5254697eecf1583aef635eb916818901dab%26scene%3D21%23wechat_redirect)、[Flash-Decoding](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650893194%26idx%3D1%26sn%3D06e0a468e4b3cd4236ee91718d87b5b9%26chksm%3D84e4aff4b39326e2015b7853009f30623db19207295159559e7126ef3fdd75fa4984c32d9d44%26scene%3D21%23wechat_redirect)的作者。FlashAttention 是一种对注意力计算进行重新排序并利用经典技术（平铺、重新计算）加快速度并将内存使用从序列长度的二次减少到线性的算法。Flash Attention v2、Flash-Decoding 都是建立在 Flash Attention 基础上的后续工作，把大模型的长文本推理效率不断推向极限。在 Mamba 之前，Tri Dao 和 Albert Gu 也有过合作。

**方法创新**



论文第 3.1 节介绍了如何利用合成任务的直觉来启发选择机制，第 3.2 节解释了如何将这一机制纳入状态空间模型。由此产生的时变 SSM 不能使用卷积，导致了高效计算的技术难题。研究者采用了一种硬件感知算法，利用当前硬件的内存层次结构来克服这一难题（第 3.3 节）。第 3.4 节描述了一个简单的 SSM 架构，不需要注意力，甚至不需要 MLP 块。第 3.5 节讨论了选择机制的一些其他特性。



**选择机制**



研究者发现了此前模型的一个关键局限：以依赖输入的方式高效选择数据的能力（即关注或忽略特定输入）。



序列建模的一个基本方法是将上下文压缩到更小的状态，我们可以从这个角度来看待当下流行的序列模型。例如，注意力既高效又低效，因为它根本没有明确压缩上下文。这一点可以从自回归推理需要明确存储整个上下文（即 KV 缓存）这一事实中看出，这直接导致了 Transformer 缓慢的线性时间推理和二次时间训练。



递归模型的效率很高，因为它们的状态是有限的，这意味着恒定时间推理和线性时间训练。然而，它们的高效性受限于这种状态对上下文的压缩程度。



为了理解这一原理，下图展示了两个合成任务的运行示例：



![img](https://pic3.zhimg.com/80/v2-9cf9a32aa82037db77e870bbb4f618fe_720w.webp)



研究者设计了一种简单的选择机制，根据输入对 SSM 参数进行参数化。这样，模型就能过滤掉无关信息，并无限期地记住相关信息。



将选择机制纳入模型的一种方法是让影响序列交互的参数（如 RNN 的递归动力学或 CNN 的卷积核）与输入相关。算法 1 和 2 展示了本文使用的主要选择机制。其主要区别在于，该方法只需将几个参数 ∆，B，C 设置为输入函数，并在整个过程中改变张量形状。这些参数现在都有一个长度维度 L ，意味着模型已经从时间不变变为时间可变。



![img](https://pic2.zhimg.com/80/v2-1ea263b01ffde3b714772452f265f4a9_720w.webp)



**硬件感知算法**



上述变化对模型的计算提出了技术挑战。所有先前的 SSM 模型都必须是时间和输入不变的，这样才能提高计算效率。为此，研究者采用了一种硬件感知算法，通过扫描而不是卷积来计算模型，但不会将扩展状态具体化，以避免在 GPU 存储器层次结构的不同级别之间进行 IO 访问。由此产生的实现方法在理论上（与所有基于卷积的 SSM 的伪线性相比，在序列长度上呈线性缩放）和现有硬件上都比以前的方法更快（在 A100 GPU 上可快达 3 倍）。



![img](https://pic3.zhimg.com/80/v2-e6da5562e29fec49347251759f53cb4e_720w.webp)



**架构**



研究者将先前的 SSM 架构设计与 Transformer 的 MLP 块合并为一个块，从而简化了深度序列模型架构，形成了一种包含选择性状态空间的简单、同质的架构设计（Mamba）。



与结构化 SSM 一样，选择性 SSM 也是一种独立的序列变换，可以灵活地融入神经网络。H3 架构是著名的同质化架构设计的基础，通常由线性注意力启发的块和 MLP（多层感知器）块交错组成。



研究者简化了这一架构，将这两个部分合二为一，均匀堆叠，如图 3。他们受到门控注意力单元（GAU）的启发，该单元也对注意力做了类似的处理。



![img](https://pic1.zhimg.com/80/v2-3ace07796cb14eebe82fc677a7eabb5c_720w.webp)



选择性 SSM 以及 Mamba 架构的扩展是完全递归模型，几个关键特性使其适合作为在序列上运行的通用基础模型的骨干：



1. 高质量：选择性为语言和基因组学等密集模型带来了强大的性能。
2. 快速训练和推理：在训练过程中，计算量和内存与序列长度成线性关系，而在推理过程中，由于不需要缓存以前的元素，自回归展开模型每一步只需要恒定的时间。
3. 长上下文：质量和效率共同提高了实际数据的性能，序列长度可达 100 万。



**实验评估**



实证验证了 Mamba 作为通用序列基础模型骨干的潜力，无论是在预训练质量还是特定领域的任务性能方面，Mamba 都能在多种类型的模态和环境中发挥作用：



合成任务。在复制和感应头等重要的语言模型合成任务上，Mamba 不仅能轻松解决，而且能推断出无限长的解决方案（>100 万 token）。



![img](https://pic2.zhimg.com/80/v2-af7288a57ed6bebeff92210115b31e1d_720w.webp)



音频和基因组学。在音频波形和 DNA 序列建模方面，Mamba 在预训练质量和下游指标方面都优于 SaShiMi、Hyena、Transformer 等先前的 SOTA 模型（例如，在具有挑战性的语音生成数据集上将 FID 降低了一半以上）。在这两种情况下，它的性能随着上下文长度的增加而提高，最高可达百万长度的序列。



![img](https://pic2.zhimg.com/80/v2-6521c151af327a8e7cb67545042a566d_720w.webp)



语言建模。Mamba 是首个线性时间序列模型，在预训练复杂度和下游评估方面都真正达到了 Transformer 质量的性能。通过多达 1B 参数的缩放规律，研究者发现 Mamba 的性能超过了大量基线模型，包括 LLaMa 这种非常强大的现代 Transformer 训练配方。



![img](https://pic4.zhimg.com/80/v2-4a0e7568e07eedb0f175eeee1d0bb5e7_720w.webp)



与类似规模的 Transformer 相比，Mamba 具有 5 倍的生成吞吐量，而且 Mamba-3B 的质量与两倍于其规模的 Transformer 相当（例如，与 Pythia-3B 相比，常识推理的平均值高出 4 分，甚至超过 Pythia-7B）。



![img](https://pic1.zhimg.com/80/v2-64b0b42fb26bf099938e3163353921c4_720w.webp)





## Reference

Hepta. “How to Judge RWKV (arXiv 2305.13048)？,” September 15, 2023. https://www.zhihu.com/question/602564718/answer/3211669817. 

