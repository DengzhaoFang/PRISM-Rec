本项目目前的核心动机可以系统地概括为：我们希望把 item 从“原始多模态 embedding”转化为“适合生成式推荐使用的 semantic ID 与配套 dense representation”，但现有 stage1 tokenizer 往往只是在做输入表示的重构与量化，而不是在做“面向推荐目标的表示提炼”。对我们这个场景来说，sentence-T5 的文本 embedding 和 LightGCN 的协同 embedding 都不是无噪声的真值，它们只是对 item 的两种不完美观测：文本侧可能包含与推荐无关的语义冗余，协同侧会受到曝光偏置、热门偏置和长尾稀疏的影响。因此，相对于 stage2 的 next-item
recommendation 目标，真正重要的不是 stage1 能否把输入 embedding 尽量原样复现，而是它能否产出一套既保持 item 唯一性与层次结构、又尽可能保留推荐判别信息的 semantic ID 和 dense embedding。换句话说，stage1 不应被理解为“纯压缩器”，而应被理解为“推荐语义的离散化前端”，其价值最终必须由 stage2 的准确率来检验，而不能只看 stage1 的重构或聚类 proxy。

基于这个动机，我们真正希望做出的创新点，是把 stage1 从“重构驱动的 tokenizer”升级为“推荐感知的多模态 tokenization 模块”，重点不只是去噪，更是有目标地融合与筛选信息。一个理想的方向是：在 stage1 中显式建模不同 item 的模态可靠性与偏好差异，使模型能够自适应地判断当前 item 更该信任文本特征、协同特征，还是两者的共享部分；同时在量化前后的表示空间中加入与 stage2 一致的结构约束，使“推荐上应当接近的 item”在 semantic ID 和 purified dense embedding 中也更接近，而“推荐上应当区分的 item”在离散码空间里更可分。
这样，semantic ID 负责为 stage2 提供稳定、唯一、可生成的离散目标，purified dense embedding 负责提供更细粒度的语义与行为补充，两者共同服务于生成式推荐，而不是各自独立优化。最终我们希望形成的是一套从 stage1 到 stage2 的一致性设计：stage1 学到的不是“最会复原输入的编码”，而是“最有利于后续生成式推荐的 item 表征”。



目前的粗糙想法：  
encoder输入之前的原始表征：由推荐感知做监督，然后俩模态交叉注意力，互补出一个融合表达（也就是说单个模态既要看另一个模态的脸色也要看推荐效果的脸色然后想办法让自己变得更好），然后输入给encoder。不要让推荐感知去强行约束z或者zq。所有约束落在encoder之前的这个输入上。


我觉得你的直觉是对的，而且方向上比我们前一版“把推荐感知压到 z / z_q 上”更合理。更准确地说，你想做的其实不是“在 tokenizer 后面补一个推荐 loss”，而是把 stage1 前端改造成一个 recommendation-aware 的多模态表征精化模块：先让文本模态和协同模态在进入encoder 之前彼此交互、彼此校正，再用推荐信号告诉这两个模态“什么信息值得保留、什么信息应该被抑制”，最后把这个更干净、更面向推荐目标的融合表达送进后续的 encoder -> RQ -> decoder 主干。这样做的核心好处是，推荐监督作用在 连续、可塑、尚未量化的输入空间，
而不是直接去拉扯已经同时受重构、量化、码本分配约束的 z / z_q；因此它更有机会避免我们之前遇到的几个潜在问题：一是推荐 loss 和 VQ/commit loss 在 latent 空间里直接抢梯度，二是 stage1 可以通过旁路“应付”推荐约束但不真正改善主路径，三是 stage1 proxy 看起来变好但 stage2 实际没有受益。你的新想法本质上是在说：不要去改量化后的结果，而是先把“喂给量化器的东西”变得更适合推荐。 这条逻辑是成立的。

如果把这个想法整理成更系统、可以写进方案里的表述，我会建议这样描述，清晰描述后的方法：
我们在 stage1 中引入一个 Recommendation-Guided Cross-Modal Refinement 模块，放在 encoder 之前。对于每个 item，先将文本 embedding 和协同 embedding 映射到同一隐空间，随后做双向交叉注意力，让文本分支从协同分支中吸收对推荐有用但自身缺失的信息，让协同分支从文本分支中吸收更稳定的语义约束；但这种“吸收”不是无条件的，而是由 item-level 的推荐监督信号来调制，例如由训练交互图中提取的 next-item 邻域、共现邻域、LightGCN 近邻或基于序列统计构造的推荐原型来提供指导。于是每个模态分支都同时受到两类信号影响：一类是来自另一模态的互补信息，另一类是来自推荐目标的有效性反馈，最终得到 refined text、refined collab 以及一个 fused input，再送入 encoder/RQ-VAE 学 semantic ID。这样，推荐感知不再直接约束z / z_q，而是专注于优化前端输入分布；RQ-VAE 仍然专注于对“精化后的 item 表征”进行稳定离散化。这个版本比“直接在 latent 上做推荐约束”更容易保证目标分工清晰、梯度关系更稳定，也更符合“先精化输入，再离散编码”的工程逻辑。唯一需要补的一点是：为了防止refinement 过度改写 item 身份，前端模块最好是 残差式、保守式更新，再配一个小的 self-anchor / drift regularization，避免它为了迎合推荐信号而把原始语义冲掉。



补充信息：
stage1代码路径：src/sid_tokenizer/prism
stage2代码路径：src/recommender/prism
stage1批量实验脚本：scripts/prism/batch/stage1.py
如果你需要运行任何代码以查看数据构成或者测试功能，可以先激活项目主目录下的source ./.venv/bin/activate虚拟环境。

我补看一下数据接口里现成能拿到哪些训练图/序列信息，好把“推荐感知信号”落到一个真正可实现、噪声可控的方案上。
• Explored
  └ Read multimodal_dataset.py
    Search train\.parquet|valid\.parquet|ItemID|UserID|next item|co-occur|sequence in prism

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• 我现在更倾向于把 item-level 推荐感知信号 定义为一句话：

  “这个 item 往往会在什么样的历史上下文之后被点击/购买。”

  如果按这个定义来选方案，我认为效果最好的主信号，不是 LightGCN KNN，也不是简单的共现图，而是：

  基于 train.parquet 的 target-side context prototype，外加 directed transition 的 backoff。

  也就是，对每个 item i，不要问“谁和它像”，而要问“哪些历史上下文会把用户带到它”。这比直接拿 KNN 或共现邻居更接近 stage2 的真实目标，因为 stage2 学的是 history -> next item，不是 item -> similar item。

  ———

  一、我认为最好的 item-level 推荐感知信号是什么

  最推荐的主方案是：

  为每个 target item 构造一个静态的 context prototype teacher。

  具体做法是，从 train.parquet 中收集所有以 i 为 target 的训练样本：

  (h1, h2, ..., hm) -> i

  对每条历史 H=(h1...hm)，先把它压成一个上下文向量 c(H)，然后把所有指向 i 的上下文向量聚合起来，得到 i 的推荐原型 p_i。

  最简单也最稳的定义是：

  c_t(H) = 历史中各 item 的 text embedding 做 recency-weighted average
  c_c(H) = 历史中各 item 的 collab embedding 做 recency-weighted average
  c_x(H) = [c_t(H), c_c(H)] 或一个轻量融合后的上下文向量

  权重不要平均，应该强调最近几步，因为 stage2 的 next-item 更受近期行为驱动：

  w_j ∝ exp(-γ * 距离target的步数)

  然后对每个 item i：

  p_i = mean_{H -> i} c_x(H)

  或者更稳一点：

  p_i = weighted_mean_{H -> i} c_x(H)

  权重可以按样本频次、session 质量、或者历史长度做轻量修正，但第一版不用太复杂。

  这个 p_i 就是你要的 item-level recommendation-aware signal。它表达的不是“i 和谁长得像”，而是“什么样的历史上下文会把用户推向 i”。

  ———

  二、为什么我觉得它比别的候选更好

  如果把几个候选信号按优先级排，我会这样排：

  1. 最好：target-side context prototype
  2. 次好：directed next-item / predecessor transition prototype
  3. 可做辅助：同序列共现 prototype
  4. 不建议做主监督：LightGCN KNN

  原因很明确。

  1. 为什么不是 LightGCN KNN

  这是我们前面最大的教训之一。LightGCN 本身就是从交互图里学出来的协同行为表示，它已经带有热门偏置、曝光偏置和稀疏噪声。你如果再拿它的 KNN 去监督 stage1，相当于让协同模态拿自己当老师，容易出现两种问题：

  - 监督是循环的：collab -> collab-neighbor -> refined collab
  - 热门 item 的行为模式会被放大，长尾 item 会更弱
  - 它天然更偏向“谁和谁在图上靠近”，不等于“什么历史会指向这个 item”

  所以 LightGCN KNN 可以留作一个弱辅助信号，或者做 confidence feature，但不适合做主 teacher。

  2. 为什么不是单纯共现图

  同序列共现比 KNN 好一点，因为它来自真实序列，但它的问题是太粗：

  - 它是对称的，不区分先后顺序
  - 它没有“谁是 target”的信息
  - 它会把“同一 session 里碰巧一起出现”也当成正信号

  而 stage2 关心的是生成 next item，这本质上是一个带方向、带局部时序的任务。只用共现，语义太宽了。

  3. 为什么 directed transition 比较好

  比如构造：

  - predecessor graph：哪些 item 经常出现在 i 之前
  - successor graph：i 之后经常出现哪些 item

  尤其 predecessor 更重要，因为它直接对应 history -> i。

  它比共现更贴近 stage2，也比 KNN 更少“自监督循环”。但它还是 pairwise 的，只看到 item-item 过渡，没看到完整历史上下文。

  4. 为什么 target-side context prototype 最好

  因为它是最接近 stage2 输入条件的静态 teacher。

  stage2 的真实任务不是：

  - “预测和 target 相似的 item”
  - 也不是“预测 target 的 graph 邻居”

  而是：

  - “给定一段历史，输出 target 的 semantic ID”

  所以如果你能把所有能导向 i 的历史上下文压成 p_i，那这个 p_i 就是最自然的 item-level 推荐监督信号。

  它的几个优点是：

  - 直接面向 history -> target 任务，不绕弯
  - 不依赖当前 stage1 的 latent，不会和 VQ 空间打架
  - 不依赖当前 stage2 的中间状态，不会形成在线共振
  - 可以完全离线预计算，训练稳定
  - 监督落在 encoder 之前的连续输入上，梯度最自然

  ———

  三、结合之前的经验，我建议的最佳落地版本

  如果让我现在定一个最稳、最可能有效的版本，我会建议：

  主信号：train.parquet 上的 target-context prototype
  辅助信号：directed predecessor transition prototype
  不把 LightGCN KNN 当主监督，只当弱先验或置信度特征

  可以写成一个分层 backoff 结构：

  teacher_i = alpha_i * p_ctx(i) + (1 - alpha_i) * p_trans(i)

  其中：

  - p_ctx(i)：所有 H -> i 的历史上下文原型
  - p_trans(i)：基于一跳 predecessor 的过渡原型
  - alpha_i：由 target item 的上下文支持度决定

  例如：

  alpha_i = n_ctx(i) / (n_ctx(i) + τ)

  这里 n_ctx(i) 是 item i 在 train 中作为 target 出现的次数。

  这就自动解决了 head/tail 问题：

  - 热门 item / 高支持 item：n_ctx(i) 大，更多信任 p_ctx(i)，因为它的真实推荐上下文很充足
  - 长尾 item / 低支持 item：n_ctx(i) 小，减少对 noisy context prototype 的依赖，更多回退到 transition 或 self-anchor

  注意，这里不是硬编码“热门更靠 collab、长尾更靠 text”，而是更合理的：

  - 谁的推荐上下文证据更可靠，就更听谁的
  - 证据不可靠时，再让文本 identity 保底

  这比我们之前的 popularity prior 更自然，也更不容易引入错误偏置。

  ———

  四、这个信号怎么进入你想要的 cross-attention refinement

  你现在的想法是：

  - 文本模态和协同模态先做交叉注意力
  - 让每个模态既看另一个模态，也看推荐目标
  - 监督全部落在 encoder 之前

  我觉得最合适的接法是：

  1. 先把 h_t, h_c 投到同一隐空间
  2. 计算离线 teacher teacher_i
  3. 用 teacher_i 去调制 cross-attention / fusion，而不是直接去监督 z / z_q

  更具体一点，可以是：

  - 文本分支在看协同分支时，不是无条件吸收，而是问：
    “协同里哪些成分更有助于让我贴近 teacher_i？”

  - 协同分支在看文本分支时，也问同样的问题
  - 最终得到 h_t_refined, h_c_refined, h_fused

  然后只在这里加推荐约束，比如：

  L_ctx = 1 - cos(h_fused, stopgrad(teacher_i))

  或者更稳一点，用 set-wise / contrastive：

  anchor = h_fused(i)
  positive = teacher_i
  negative = teacher_j (j != i)

  但我更建议一开始先用 cosine regression / cosine alignment + small contrastive，不要上来就全 InfoNCE。

  因为我们前面的经验说明：一旦把“推荐结构损失”做得太强、太激进，就容易把主干训练带歪。

  ———

  五、基于前面失败经验，我会明确避开的坑

  这部分很重要。

  1. 不要让 teacher 在线依赖当前模型输出

  不要搞成“当前 refined 表示再反过来更新 teacher”。这会不稳定，容易自激。

  第一版最好是：

  - teacher_i 完全离线预计算
  - 训练时 stop_grad
  - 如果要更新，也是一整个 epoch 或若干 epoch 后整体刷新一次

  2. 不要直接用 z / z_q 承接推荐监督

  这正是我们之前方案最容易冲突的点。你现在这个新想法比之前更好，就是因为把监督前移到了 pre-encoder input。

  3. 不要把所有正样本关系都做成“邻居拉近”

  对 item-level supervision 来说，最安全的是“朝 teacher prototype 对齐”，而不是“把所有上下文相关 item 两两都拉近”。后者太容易过平滑，最后 semantic ID 变糊。

  4. 不要让 LightGCN teacher 成为主导

  否则很容易又走回“协同模态自我强化”的老路，最后文本只剩陪跑。

