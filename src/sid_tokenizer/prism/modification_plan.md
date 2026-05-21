请你使用Superpowers的相关技能，对prism的tokenizer模型进行相关模块的修改和优化，工作目录主要是src/sid_tokenizer/prism。
在开发前，先创建一个新的开发分支。如果你需要运行任何代码以查看数据构成或者测试功能，请先运行source ./.venv/bin/activate以激活虚拟环境。

### 详细说明：
对于RQ-VAE 作为SID tokenizer，其输入有两个来源，一个是协同embedding 64维，一个是文本embedding 768维。相关数据文件夹：dataset。
下面是一些宏观的实现方案，注意，下面的方案只作为整体思路的指导，还有更多工程上的、代码上的实现细节是需要你来把控的，务必要确保代码优雅、可用，绝对不可以在未测试成功的情况下声称完成任务。所有目的是为了tokenizer阶段的信息去噪、特征整合，从而得到表征质量更高的item SID。

#### 模块一：信息密度对齐（Information Density Equalization, IDE）

不直接操作原始维度，先将文本（语义冗余）进行压缩，将协同（信息密集但维度低）进行空间展开。


$$h_t = \text{LayerNorm}(W_{t} e_t) \quad W_t \in \mathbb{R}^{d \times 768}$$

$$h_c = \text{LayerNorm}(W_{c} e_c) \quad W_c \in \mathbb{R}^{d \times 64}$$


设定 $d = 128$。引入 LayerNorm 确保两个分布在相同的数值尺度内，为计算一致性打下基础。

#### 模块二：跨模态互校验去噪（Mutual Cross-modal Denoising, MCD）

彻底抛弃流行度，完全依赖两个模态之间的关系来进行非对称去噪。
首先，计算 Item 级别的跨模态一致性标量：


$$s = \frac{1}{2}(\cos(h_t, h_c) + 1)$$


**操作 A：对协同特征进行“可靠性”去噪（用一致性指导）**
行为数据有噪音（误点等），当它和文本不一致时（$s$ 较低），协同特征不可信。


$$g_c = \sigma(W_{gc} [h_c \parallel s] + b_{gc})$$


引入类似残差的安全回退机制（Safety Fallback），当协同高度不可信时，回退到稳定的文本语义：


$$\hat{h}_c = g_c \odot h_c + (1 - g_c) \odot h_t$$

**操作 B：对文本特征进行“相关性”降权（用协同信号指导）**
文本包含与推荐无关的形容词。我们用协同特征 $h_c$ 作为 Context，生成一个门控来抑制 $h_t$ 中的冗余维度。


$$g_t = \sigma(W_{gt} h_c + b_{gt})$$

$$\hat{h}_t = g_t \odot h_t$$

#### 模块三：密度均衡融合（Density-Balanced Fusion）

经过上述互校验净化后，我们将两者拼接。由于此时双方都是 128 维且去除了各自的冗余/噪声，梯度的流动将是均衡的。


$$z = \text{Encoder}([\hat{h}_c \parallel \hat{h}_t])$$


这个 $z \in \mathbb{R}^{256}$ 将输入给下游的 RQ-VAE 去生成 Semantic IDs。

#### 模块四：【核弹级创新】序列感知对比重构（Sequence-Aware Contrastive Objective, SACO）

为了回答此问题（如何让第一阶段的去噪与第二阶段的序列推荐目标挂钩）。
如果在第一阶段，RQ-VAE 的损失函数仅仅是双头重构损失（DHR），那么模型依然是在无脑拟合原始带噪特征。
我们需要引入一个**序列对比损失 $\mathcal{L}_{SAC}$**：
如果 Item A 和 Item B 在真实用户的历史序列中**紧密相连（Co-occurrence）**，那么在第一阶段，经过去噪融合后的隐表示 $z_A$ 和 $z_B$ 的距离应该被拉近；如果不共现，距离被推远。


$$\mathcal{L}_{SAC} = - \sum_{(A,B) \in \mathcal{S}_{pos}} \log \frac{\exp(\cos(z_A, z_B)/\tau)}{\sum_{N \in \mathcal{S}_{neg}} \exp(\cos(z_A, z_N)/\tau)}$$

* $\mathcal{S}_{pos}$ 代表在同一个 Session/用户历史中相邻出现的 Item 对。
* $\mathcal{S}_{neg}$ 代表同一 Batch 内随机采样的负样本对。

**Stage 1 最终的 Loss 函数：**


$$\mathcal{L}_{stage1} = \mathcal{L}_{DHR} + \beta \mathcal{L}_{commit} + \lambda_{sac} \mathcal{L}_{SAC}$$


*(注意：这里彻底移除了原来的 $\mathcal{L}_{ACD}$ 和 $\mathcal{L}_{HSA}$)*

---

### 确保这套方案一定能赢审稿人

1. 用“跨模态一致性”取代了被诟病的“流行度置信度”，且引入了针对文本和协同各自缺陷的“非对称互校验去噪”。这是目前多模态表征学习最前沿的思路。
2. 通过信息密度对齐（IDE）和固定维度的独立门控，从数学上杜绝了 768 维梯度压倒 64 维梯度的隐患。
3. 引入 $\mathcal{L}_{SAC}$ 是画龙点睛之笔。你可以在文中宣称：“将序列推荐任务的全局共现信号（Global Co-occurrence）作为自监督信号注入到了 Semantic ID 的量化去噪阶段，彻底打破了量化阶段与生成阶段的任务割裂”。这直接升华了这篇论文的立意。
4. 第二阶段无需改动：依然可以完美复用基于 MoE 的 Dynamic Semantic Integration (DSI)。第一阶段产出的 SID 更纯净，第二阶段 MoE 恢复连续特征时的输入底座也更扎实。

注意，确保有相关的超参数控制MCD和SACO这两个模块的开关，方便后续做消融实验。






### 一、 助手工作评估与训练现状分析

从训练日志来看，所有的指标不仅正常，而且**状态极其健康，可以说是大获成功**。

#### 1. 为什么说目前的状态极其健康？

* **UPR (Unified Preference Reconstruction) 完美收敛**：从 0.715 稳步降到 0.608。助手把原来两头拉扯的双解码器（768D+64D）重构成了一个单一的 256D 解码器，并使用 `z_clean.detach()` 作为重构目标。这是一个**神来之笔**。如果不 detach，网络会走捷径把 MCD 门控全部置 0 来降低 MSE 损失；加上 detach 后，解码器就被迫老老实实地去逼近净化后的表征。
* **Perplexity（码本利用率）极高**：Layer 1/2/3 的 Perplexity 分别稳定在 204, 218, 218 左右（假设你的码本大小是 256）。这意味着**几乎 85% 以上的 Codebook 都被有效激活了！** 在很多 RQ-VAE 论文里，最头疼的就是 Codebook Collapse（只用了几十个码）。由于你的 MCD 去除了冗余和噪声，并做了信息密度均衡（IDE），RQ-VAE 吃到了极其干净的高方差输入，聚类效果绝佳。
* **MCD Consistency 的方差证明了“自适应去噪”正在工作**：日志显示 `mean=0.493, std=0.045, range=[0.36, 0.62]`。这证明模型**并没有把一致性权重变成常数**。对于高噪 Item（一致性 0.36），模型切断了协同信号；对于高质量 Item（一致性 0.62），模型充分融合。

#### 2. MCD 去噪模块有损失函数吗？

**没有显式的损失函数，它是通过“信息瓶颈（Information Bottleneck）”和下游梯度来隐式学习的。**
在助手的设计下，MCD 模块的梯度来源只有两条：

1. **SACO (序列感知对比损失)**：为了让共现（Co-occurrence）的 Item 距离拉近，梯度回传时会强迫 MCD 模块“打开对特征有益的门控，关掉导致 Item 距离拉远的随机噪声”。
2. **RQ-VAE 的 Commitment Loss**：强迫进入潜空间的向量分布紧凑。
这种无监督、任务驱动的门控机制，比你之前强加的“流行度监督损失（GateSupervisionLoss）”要高级和安全得多，审稿人完全无法在这点上攻击你。