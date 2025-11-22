论文《Learnable Item Tokenization for Generative Recommendation (LETTER)》

LETTER的核心逻辑在于：现有的Codebook-based（如TIGER）方法虽然引入了语义，但**缺少协同信号（Collaborative Signals）**且存在**Code分配偏差（Code Assignment Bias）**。

以下是复现需求的详细技术解析：

### 1. 核心方法详解：三大正则化 (The Three Regularizations)

LETTER 本质上是一个**可学习的Tokenizer**，它基于RQ-VAE（Residual Quantized VAE）架构，但在训练过程中引入了额外的损失函数来约束Code的生成。

#### A. 语义ID生成过程 (Item Tokenization) 的创新

在这一步，目标是将Item $i$ 转化为离散的Token序列 $c = [c_1, c_2, \dots, c_L]$。LETTER在标准的RQ-VAE重构损失基础上，增加了两项关键改进：

**1. 协同正则化 (Collaborative Regularization) - 解决语义与协同信号不对齐**
* [cite_start]**问题：** 语义相似的物品（如两个吉他）可能有完全不同的交互模式（用户群不同）。仅靠语义生成的Code无法反映这种协同差异 [cite: 109, 199]。
* **改进：** 引入对比学习，强制RQ-VAE生成的**量化嵌入 (Quantized Embedding)** $\hat{z}$ 与预训练的**CF嵌入 (CF Embedding)** $h$ 对齐。
* **实现：** 使用一个训练好的CF模型（如SASRec或LightGCN）作为Teacher，冻结其参数，提取物品的CF Embedding。
* **公式：**
    $$\mathcal{L}_{CF} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\langle \hat{z}_i, h_i \rangle)}{\sum_{j=1}^{B} \exp(\langle \hat{z}_i, h_j \rangle)}$$
    [cite_start]其中 $\hat{z} = \sum_{l=1}^{L} e_{c_l}$ 是RQ-VAE各层Code Embedding的和 [cite: 248]。

**2. 多样性正则化 (Diversity Regularization) - 解决Code分配偏差**
* [cite_start]**问题：** RQ-VAE倾向于通过“马太效应”分配Code，导致某些Code被过度使用，而长尾Code被忽略，这会导致生成时的偏差 [cite: 201, 254]。
* **改进：** 强制Codebook中的Code Embedding在表示空间中均匀分布。
* **实现：** 对每一层Codebook $C_l$ 中的Embedding进行K-means聚类（分为$K$组）。然后使用对比损失：拉近同簇内的Code Embedding，推远不同簇的。
* **公式：**
    $$\mathcal{L}_{Div} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\langle e_{c_l}^i, e_+ \rangle)}{\sum_{j=1}^{N-1} \exp(\langle e_{c_l}^i, e_j \rangle)}$$
    [cite_start]这促使Codebook在潜空间各向同性分布，提高Code利用率 [cite: 259, 410]。

#### B. 序列推荐模型 (Generative Model) 的创新

在下游任务中（使用Transformer/LLM进行Next Item预测），LETTER主要改进了损失函数。

**1. 排名引导的生成损失 (Ranking-guided Generation Loss)**
* [cite_start]**问题：** 传统的生成式损失（Cross Entropy / NLL）只关注预测准确性，忽略了Top-K推荐所需的排名能力，对Hard Negative挖掘不足 [cite: 271]。
* **改进：** 引入温度系数 $\tau$ (Temperature) 修改Softmax损失。
* [cite_start]**原理：** 较小的 $\tau$ 会放大Hard Negative的惩罚权重。论文证明了最小化该损失等价于优化 OPAUC (One-way Partial AUC)，这与Recall和NDCG强相关 [cite: 278, 589]。
* **公式：**
    $$\mathcal{L}_{rank} = -\sum_{t=1}^{|y|} \log \frac{\exp(p(y_t)/\tau)}{\sum_{v \in V} \exp(p(v)/\tau)}$$
    [cite_start]其中 $V$ 是词表（包含Code tokens）[cite: 274]。

---

### 2. 复现指南：详细设计与参数清单

基于论文实验部分（Section 4.1.3 & 4.3.5），以下是你复现代码所需的具体参数配置。

#### 第一阶段：训练 Tokenizer (LETTER)


* **Loss Weights (Eq. 5):**
    * [cite_start]Semantic Regularization ($\mathcal{L}_{Sem}$): 权重 1.0 (默认)。包含 RQ-VAE的 $\beta_{commit} = 0.25$ (用于平衡encoder和codebook更新) [cite: 321]。
    * [cite_start]Collaborative Regularization ($\alpha$): **0.02** (推荐值) [cite: 517]。
    * [cite_start]Diversity Regularization ($\beta$): **1e-4** (推荐范围在1e-3到1e-5之间，文中图表显示小数值即可生效) [cite: 321, 519]。
* **Diversity Clustering:**
    * [cite_start]Cluster Number ($K$): **10** (使用Constrained K-means) [cite: 319, 521]。

    除了特别指明的超参数，其余的时候和tiger一致的超参数以确保公平。

#### 第二阶段：训练生成式推荐模型 (Downstream Model)

* **Ranking Loss Temperature ($\tau$):**
    * [cite_start]推荐范围: **0.8 ~ 1.0** (图8显示在0.8左右性能较好，过小会导致False Negative问题) [cite: 509, 525]。
* [cite_start]**Inference:** 使用 Trie-based constrained generation (前缀树约束生成) 保证生成的Token序列在Codebook中有效 [cite: 285]。

    除了特别指明的超参数，其余的时候和tiger一致的超参数以确保公平。

