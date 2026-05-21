实验二：基于活跃度分桶的序列预测测试（验证协同的“可靠性噪声”）

实验目的：证明协同特征虽然贴合偏好，但极不稳定，高度依赖交互密度；而文本特征虽然死板，但在稀疏场景下具备兜底能力。

实验做法：不用复杂的生成模型，仅用一个最基础的序列模型（如 SASRec）。
分别单独使用纯文本特征（Text-only）和纯协同特征（Collab-only）作为 Item Embedding 进行训练。
在测试集上，将目标 Item 按照交互频率（Popularity / Node Degree）分为 5-10 个桶（从 Long-tail 到 Popular）。
分别记录两个模型在不同分桶上的 Recall@10。

出图建议（Figure 2）：X轴：Item 流行度分桶（Long-tail $\rightarrow$ Popular）。Y轴：Recall@10 性能。

预期视觉呈现：Collab-only 曲线：呈现极其陡峭的上升趋势。在尾部（Long-tail）性能极低，在头部（Popular）性能极高。Text-only 曲线：呈现一条相对平缓的水平线。在尾部击败协同，但在头部被协同碾压。预期图注（Caption）撰写：Figure 1: Performance comparison across item popularity groups. Collaborative signals exhibit severe reliability noise in sparse scenarios (long-tail), whereas text semantics remain robust but sub-optimal due to structural rigidity.