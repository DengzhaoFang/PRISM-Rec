要复现 **ActionPiece** (Contextually Tokenizing Action Sequences for Generative Recommendation)，你需要从**数据预处理**、**Tokenizer（核心创新点）**、以及**Recommender（生成式推荐模型）** 这三个主要部分入手。

[cite_start]以下是基于论文原文 [cite: 1, 12, 120, 668, 679, 701, 714, 815, 904] 整理的复现指南。

---

### 第一部分：数据预处理 (Data Preparation)

在构建 Tokenizer 之前，必须先将用户行为序列转化为“特征集合序列” (Sequence of Feature Sets)。

1.  **特征提取与量化 (Feature Engineering):**
    * [cite_start]**文本特征:** 将商品的 Title, Price, Brand, Category, Description 等文本拼接，使用 `sentence-t5-base` 编码得到 768 维向量 [cite: 846-848]。
    * **量化 (Quantization):** 使用 OPQ (Optimized Product Quantization) 将向量量化。
        * [cite_start]**参数:** 每个 Item 量化为 **4 个 code** (即 $m=4$)，每个 code 的 codebook 大小为 **256** [cite: 853-855]。
        * **结果:** 每个 Item 被表示为一个无序的特征集合 $\mathcal{A}_j = \{f_{j,1}, f_{j,2}, f_{j,3}, f_{j,4}\}$。
2.  **序列构建:**
    * 按时间顺序排列用户的交互 Item。
    * [cite_start]**截断:** 序列最大长度设为 **20** [cite: 843]。
    * [cite_start]**格式:** 输入数据即为 $S' = \{\mathcal{A}_1, \mathcal{A}_2, ..., \mathcal{A}_t\}$ [cite: 131]。

---

### 第二部分：Tokenizer 复现 (核心难点)

ActionPiece 的 Tokenizer 类似于 BPE，但它是针对“集合序列”设计的，包含**词表构建 (Training)** 和 **分词 (Segmentation)** 两个步骤。

#### 1. 词表构建 (Vocabulary Construction)
你需要实现一个自定义的训练算法（见论文 Algorithm 1, 2, 3 及 Figure 7），不能直接调用 `sentencepiece`。

* **初始化:**
    * [cite_start]初始词表 $\mathcal{V}_0$ 包含所有唯一的 OPQ 特征码 [cite: 155]。
* **加权共现计数 (Weighted Co-occurrence Counting) - 关键点:**
    * [cite_start]在统计 Token Pair $(c_u, c_v)$ 频率时，不同位置权重不同 [cite: 180-186]：
        * **同一个集合内 (In-set):** 权重为 $2 / |\mathcal{A}_k|$ (对于 $m=4$，权重为 $2/4=0.5$)。
        * **相邻集合间 (Adjacent sets):** 权重为 $1 / (|\mathcal{A}_k| \times |\mathcal{A}_{k+1}|)$ (对于 $m=4$，权重为 $1/16$)。
        * **目的:** 让模型捕捉上下文关系（Context-aware）。
* **高效实现策略 (Efficient Implementation):**
    * [cite_start]**数据结构:** 使用**双向链表**存储序列，使用**倒排索引 (Inverted Index)** 记录 Token Pair 出现的位置，使用 **Max-Heap (带 Lazy Update 机制)** 维护最高频 Pair [cite: 683-693]。
    * **合并更新 (Update):**
        * [cite_start]当合并两个 Token 时，如果它们来自相邻的 Action 节点，需要插入一个 **中间节点 (Intermediate Node)** [cite: 201, 204]。
        * [cite_start]中间节点用于存储跨 Action 的 Token，确保上下文信息被正确编码 [cite: 209]。
    * [cite_start]**循环:** 迭代合并频率最高的 Pair，直到词表大小达到目标值 (如 40k) [cite: 149]。

#### 2. 分词 (Segmentation via SPR)
[cite_start]训练好合并规则 (Merge Rules) 后，使用 **集合排列正则化 (Set Permutation Regularization, SPR)** 进行分词 [cite: 250, 714]。

* **步骤:**
    1.  [cite_start]对于序列中的每个特征集合 $\mathcal{A}_i$，**随机打乱**其内部特征顺序 [cite: 252]。
    2.  [cite_start]将打乱后的集合展平 (Flatten) 拼接成一个长的一维 Token 序列 [cite: 253]。
    3.  [cite_start]应用训练好的 Merge Rules 进行合并（类似标准 BPE 分词）[cite: 254]。
* [cite_start]**作用:** 这种随机性是一种天然的数据增强，使得同一个 Action 在不同 Epoch 可以产生不同的 Token 序列 [cite: 261]。

---

### 第三部分：Recommender 复现

Recommender 部分是一个标准的 Encoder-Decoder Transformer (类似 T5)，主要区别在于训练和推理的策略。

#### 1. 模型架构 (Architecture)
* [cite_start]**Backbone:** Transformer Encoder-Decoder [cite: 260, 912]。
* [cite_start]**参数配置 (以 Sports/Beauty 数据集为例) [cite: 905, 914-915]:**
    * `num_layers`: **4** (Encoder 和 Decoder 各 4 层)
    * `num_heads`: **6**
    * `d_model` (embedding dim): **128** (CDs 数据集为 256)
    * `d_ff` (feed-forward dim): **1024** (CDs 数据集为 2048)
    * `d_kv`: **64**
    * `dropout`: **0.1**

#### 2. 训练流程 (Training)
* [cite_start]**动态增强:** 在每个 Epoch 中，对输入序列 $S'$ 重新进行 SPR 分词，生成新的 Token 序列 $C_{in}$ [cite: 261]。
* [cite_start]**目标 (Target):** 模型预测下一个 Item 的特征。论文指出训练时 Target 使用原始特征序列 (original item features)，不进行 Token Merging 或增强 [cite: 927]。
* [cite_start]**超参数 [cite: 905, 919-921]:**
    * `Optimizer`: AdamW
    * `Learning Rate`: **0.005** (Sports) 或 **0.001** (Beauty/CDs)
    * `Batch Size`: **256**
    * `Warmup Steps`: **10,000**
    * `Weight Decay`: **0.15** (Sports/Beauty)
    * `Max Epochs`: 200 (带 Early Stop, patience=20)

#### 3. 推理流程 (Inference)
* **集成 (Ensembling):**
    * [cite_start]对于每个测试用例，使用 SPR 随机分词 **$q$ 次** (论文推荐 **$q=5$**) [cite: 264, 918]。
    * 将这 $q$ 个序列分别输入模型。
* **生成:**
    * [cite_start]使用 **Beam Search**，Beam Size = **50** [cite: 917]。
    * [cite_start]对 $q$ 个输出列表的得分进行平均 (Average scores)，得到最终推荐列表 [cite: 270]。

---

### 复现步骤总结清单

1.  **准备数据:** 生成 OPQ 特征 (4 codes/item)，构建序列。
2.  **编写 Tokenizer 代码:**
    * 实现带权重的共现统计 (注意 Figure 2 中的公式)。
    * 实现带 Intermediate Node 的链表更新逻辑 (Figure 3)。
    * 实现 Lazy Update Heap 以加速训练 (Figure 7)。
    * 训练得到 Merge Rules (目标词表 40k)。
3.  **编写 Dataloader:** 实现 `__getitem__`，在其中动态调用 SPR 算法 (Algorithm 4) 进行随机分词。
4.  **搭建模型:** 使用 PyTorch 或 HuggingFace 实现 4 层 T5 架构。
5.  **训练:** 跑 200 个 Epoch，确保 Loss 收敛。
6.  **推理:** 实现 $q=5$ 的 Test-time Augmentation，合并 Beam Search 结果。

[cite_start]**关键提示:** ActionPiece 的核心在于 Tokenizer 能够利用“集合”的无序性通过“中间节点”捕捉跨 Action 的上下文。如果忽略了加权统计或中间节点的设计，性能将大打折扣 [cite: 382-385]。