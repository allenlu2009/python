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

自 2017 年被提出以來，Transformer 已經成爲 AI 大模型的主流架構，但隨着模型規模的擴展和需要處理的序列不斷變長，Transformer 的侷限性也逐漸凸顯。一個很明顯的缺陷是：Transformer 模型中自注意力機制的計算量會隨着上下文長度的增加呈平方級增長，比如上下文增加 32 倍時，計算量可能會增長 1000 倍，計算效率非常低。

爲了克服這些缺陷，研究者們開發出了很多注意力機制的高效變體，但這往往以犧牲其有效性特爲代價。到目前爲止，這些變體都還沒有被證明能在不同領域發揮有效作用。幾個例子: RetNet, RWKV (Receptance Weighted Key Value).







### RNN/LSTM

* Training 長 sequence 的困難:  (1) 梯度消失; (2)  recurrent 結構所以無法在時間維度平行訓練。但在 batch 方向仍然可以平行。 
* Inference (batch=1):  雖然是 recurrent generation 無法像 CNN 可以平行展開。所有生成的 token 都 depend on 之前 tokens.  **但好處只需要前一個 time step 的 hidden information + input (Markovian)!**    **因爲訓練長 sequence 困難，另一個缺點是 attention scope 不夠！比較久之前的 tokens attention 會消失**，因此常用於語音。



### Transformer

* Training:  可以在時間維度 (token sequence) 平行訓練 (類似 prompt mode).  這是最大的優點。
* Inference (batch=1, generative mode):  (1) 好處是 attention scope 包含所有之前 context 範圍的 tokens (1K/4K/8K); (2) 缺點是 attention matrix 的計算和存儲都和 context length 的平方成正比。另一個缺點是 token generation 仍然是 recurrent.   

Transformer 徹底改變了幾乎所有[自然語言處理](https://www.zhihu.com/search?q=自然語言處理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3211669817}) (NLP) 任務，但其內存和計算複雜性卻與序列長度呈二次方關係。相比之下，RNN 和 LSTM 在內存和計算要求方面表現出[線性擴展](https://www.zhihu.com/search?q=線性擴展&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3211669817})，但由於並行化和可擴展性的限制，很難達到與 Transformer 相同的性能。



### RWKV

一種新穎的模型架構，即接收加權鍵值（Receptance Weighted Key Value, **RWKV**），它將 Transformer 的高效可並行訓練與 RNN 的高效推理相結合。我們的方法利用**線性注意力**機制，允許我們將模型制定爲 Transformer 或 RNN，它在訓練過程中並行計算，並在推理過程中保持恆定的計算和內存複雜性，從而使第一個非 Transformer 架構擴展到數十個數十億個參數。**我們的實驗表明，RWKV 的性能與類似大小的 Transformer 相當，這表明未來的工作可以利用這種架構來創建更高效的模型。**這項工作在協調序列處理任務中計算效率和模型性能之間的權衡方面邁出了重要一步。

缺點:  雖然是綫性。但是非綫性計算非常複雜？看起來是用計算換綫性 attention?





### Mamba (S4 -> S6)

最近，一項名爲「Mamba」的研究似乎打破了這一局面。這篇論文的作者只有兩位，一位是卡內基梅隆大學機器學習系助理教授 Albert Gu，另一位是 Together.AI 首席科學家、普林斯頓大學計算機科學助理教授（即將上任）Tri Dao。

一個重要創新是引入了一個名爲「選擇性 SSM」的架構，該架構是 Albert Gu 此前主導研發的 S4 架構（Structured State Spaces for Sequence Modeling ，用於序列建模的結構化狀態空間）的一個簡單泛化，可以有選擇地決定關注還是忽略傳入的輸入。一個「小小的改變」—— 讓某些參數成爲輸入的函數，結果卻非常有效。



值得一提的是，S4 是一個非常成功的架構。此前，它成功地對 Long Range Arena (LRA) 中的長程依賴進行了建模，併成爲首個在 Path-X 上獲得高於平均性能的模型。更具體地說，S4 是一類用於深度學習的序列模型，與 RNN、CNN 和經典的狀態空間模型（State Space Model，SSM）廣泛相關。SSM 是獨立的序列轉換，可被整合到端到端神經網絡架構中（ SSM 架構有時也稱 SSNN，它與 SSM 層的關係就像 CNN 與線性卷積層的關係一樣）。Mamba 論文也討論了一些著名的 SSM 架構，比如 Linear attention、H3、Hyena、RetNet、RWKV，其中許多也將作爲論文研究的基線。Mamba 的成功讓 Albert Gu 對 SSM 的未來充滿了信心。

Tri Dao 則是 [FlashAttention](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650848429%26idx%3D4%26sn%3D4665869919c379023b1bdb29568cdb2c%26chksm%3D84e578d3b392f1c59b3e4c9b986a522f6534e894800e12f5df52964d1ec45f6c067a73b53bd9%26scene%3D21%23wechat_redirect)、[Flash Attention v2](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650884509%26idx%3D3%26sn%3D65476dbc71ca235155734ed6cf52197d%26chksm%3D84e48de3b39304f54bc222ce6da480ef5ddd8874b5254697eecf1583aef635eb916818901dab%26scene%3D21%23wechat_redirect)、[Flash-Decoding](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650893194%26idx%3D1%26sn%3D06e0a468e4b3cd4236ee91718d87b5b9%26chksm%3D84e4aff4b39326e2015b7853009f30623db19207295159559e7126ef3fdd75fa4984c32d9d44%26scene%3D21%23wechat_redirect)的作者。FlashAttention 是一種對注意力計算進行重新排序並利用經典技術（平鋪、重新計算）加快速度並將內存使用從序列長度的二次減少到線性的算法。Flash Attention v2、Flash-Decoding 都是建立在 Flash Attention 基礎上的後續工作，把大模型的長文本推理效率不斷推向極限。在 Mamba 之前，Tri Dao 和 Albert Gu 也有過合作。

**方法創新**



論文第 3.1 節介紹瞭如何利用合成任務的直覺來啓發選擇機制，第 3.2 節解釋瞭如何將這一機制納入狀態空間模型。由此產生的時變 SSM 不能使用卷積，導致了高效計算的技術難題。研究者採用了一種硬件感知算法，利用當前硬件的內存層次結構來克服這一難題（第 3.3 節）。第 3.4 節描述了一個簡單的 SSM 架構，不需要注意力，甚至不需要 MLP 塊。第 3.5 節討論了選擇機制的一些其他特性。



**選擇機制**



研究者發現了此前模型的一個關鍵侷限：以依賴輸入的方式高效選擇數據的能力（即關注或忽略特定輸入）。



序列建模的一個基本方法是將上下文壓縮到更小的狀態，我們可以從這個角度來看待當下流行的序列模型。例如，注意力既高效又低效，因爲它根本沒有明確壓縮上下文。這一點可以從自迴歸推理需要明確存儲整個上下文（即 KV 緩存）這一事實中看出，這直接導致了 Transformer 緩慢的線性時間推理和二次時間訓練。



遞歸模型的效率很高，因爲它們的狀態是有限的，這意味着恆定時間推理和線性時間訓練。然而，它們的高效性受限於這種狀態對上下文的壓縮程度。



爲了理解這一原理，下圖展示了兩個合成任務的運行示例：



![img](https://pic3.zhimg.com/80/v2-9cf9a32aa82037db77e870bbb4f618fe_720w.webp)



研究者設計了一種簡單的選擇機制，根據輸入對 SSM 參數進行參數化。這樣，模型就能過濾掉無關信息，並無限期地記住相關信息。



將選擇機制納入模型的一種方法是讓影響序列交互的參數（如 RNN 的遞歸動力學或 CNN 的卷積核）與輸入相關。算法 1 和 2 展示了本文使用的主要選擇機制。其主要區別在於，該方法只需將幾個參數 ∆，B，C 設置爲輸入函數，並在整個過程中改變張量形狀。這些參數現在都有一個長度維度 L ，意味着模型已經從時間不變變爲時間可變。



![img](https://pic2.zhimg.com/80/v2-1ea263b01ffde3b714772452f265f4a9_720w.webp)



**硬件感知算法**



上述變化對模型的計算提出了技術挑戰。所有先前的 SSM 模型都必須是時間和輸入不變的，這樣才能提高計算效率。爲此，研究者採用了一種硬件感知算法，通過掃描而不是卷積來計算模型，但不會將擴展狀態具體化，以避免在 GPU 存儲器層次結構的不同級別之間進行 IO 訪問。由此產生的實現方法在理論上（與所有基於卷積的 SSM 的僞線性相比，在序列長度上呈線性縮放）和現有硬件上都比以前的方法更快（在 A100 GPU 上可快達 3 倍）。



![img](https://pic3.zhimg.com/80/v2-e6da5562e29fec49347251759f53cb4e_720w.webp)



**架構**



研究者將先前的 SSM 架構設計與 Transformer 的 MLP 塊合併爲一個塊，從而簡化了深度序列模型架構，形成了一種包含選擇性狀態空間的簡單、同質的架構設計（Mamba）。



與結構化 SSM 一樣，選擇性 SSM 也是一種獨立的序列變換，可以靈活地融入神經網絡。H3 架構是著名的同質化架構設計的基礎，通常由線性注意力啓發的塊和 MLP（多層感知器）塊交錯組成。



研究者簡化了這一架構，將這兩個部分合二爲一，均勻堆疊，如圖 3。他們受到門控注意力單元（GAU）的啓發，該單元也對注意力做了類似的處理。



![img](https://pic1.zhimg.com/80/v2-3ace07796cb14eebe82fc677a7eabb5c_720w.webp)



選擇性 SSM 以及 Mamba 架構的擴展是完全遞歸模型，幾個關鍵特性使其適合作爲在序列上運行的通用基礎模型的骨幹：



1. 高質量：選擇性爲語言和基因組學等密集模型帶來了強大的性能。
2. 快速訓練和推理：在訓練過程中，計算量和內存與序列長度成線性關係，而在推理過程中，由於不需要緩存以前的元素，自迴歸展開模型每一步只需要恆定的時間。
3. 長上下文：質量和效率共同提高了實際數據的性能，序列長度可達 100 萬。



**實驗評估**



實證驗證了 Mamba 作爲通用序列基礎模型骨幹的潛力，無論是在預訓練質量還是特定領域的任務性能方面，Mamba 都能在多種類型的模態和環境中發揮作用：



合成任務。在複製和感應頭等重要的語言模型合成任務上，Mamba 不僅能輕鬆解決，而且能推斷出無限長的解決方案（>100 萬 token）。



![img](https://pic2.zhimg.com/80/v2-af7288a57ed6bebeff92210115b31e1d_720w.webp)



音頻和基因組學。在音頻波形和 DNA 序列建模方面，Mamba 在預訓練質量和下游指標方面都優於 SaShiMi、Hyena、Transformer 等先前的 SOTA 模型（例如，在具有挑戰性的語音生成數據集上將 FID 降低了一半以上）。在這兩種情況下，它的性能隨着上下文長度的增加而提高，最高可達百萬長度的序列。



![img](https://pic2.zhimg.com/80/v2-6521c151af327a8e7cb67545042a566d_720w.webp)



語言建模。Mamba 是首個線性時間序列模型，在預訓練複雜度和下游評估方面都真正達到了 Transformer 質量的性能。通過多達 1B 參數的縮放規律，研究者發現 Mamba 的性能超過了大量基線模型，包括 LLaMa 這種非常強大的現代 Transformer 訓練配方。



![img](https://pic4.zhimg.com/80/v2-4a0e7568e07eedb0f175eeee1d0bb5e7_720w.webp)



與類似規模的 Transformer 相比，Mamba 具有 5 倍的生成吞吐量，而且 Mamba-3B 的質量與兩倍於其規模的 Transformer 相當（例如，與 Pythia-3B 相比，常識推理的平均值高出 4 分，甚至超過 Pythia-7B）。



![img](https://pic1.zhimg.com/80/v2-64b0b42fb26bf099938e3163353921c4_720w.webp)





## Reference

Hepta. “How to Judge RWKV (arXiv 2305.13048)？,” September 15, 2023. https://www.zhihu.com/question/602564718/answer/3211669817. 

