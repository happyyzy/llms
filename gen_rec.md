# 生成式推荐模型小结

# one-rec V1
1.语义id（输出头格式）:先通过预训练模型得到embeddings，再rq-kmens残差量化（做了类别均衡，不然容易出现多item对应相同id），类似于MacGNN的预处理
2.特征工程（输入头格式）：分为四类
用户静态特征
短期行为序列
正反馈(有效观看)序列
终身行为序列
最难的是终身行为序列，采样hereachy聚类来降低长序列长度+qformer得到长为2的embedding
3.预训练就是next token CE
4.后训练SFT和RL同时进行
reward的model是多塔结构，奖励分三类
1.BCE预测多个二分类指标，最后合在一起得到P-score
2.格式奖励，防止语义id找不到视频
3.业务奖励，比如有害信息等等
SFT类似于课程学习，格式和预训练一样，只是数据采样观看时长较长的，并搭配学习率退火
RL是grpo变体ECPO（Early Clipped GRPO），去掉了kL约束，因为和SFT同时进行保证了稳定性，裁剪负优势样本的梯度

reward mosel起到精排作用

# one-rec V2
1.样本组织：对每个用户的历史存成cache，而不是每条历史都作为样本，后者训练浪费大且有时序混乱问题
2.lazydecoder:样本组织实现为Context Processor，输出为kv cache，但是这里的kv是算好的，不像encoder还有kv的权重矩阵，
后面的 lazer decoder再做cross aten的时候只需直接query,这样消除了跨层的矩阵计算，提高效率。
3.强化学习
GBPO(Gradient-Bounded Policy Optimization)
改变了裁剪方式，以及设计了不需要RM的reward
技术报告里做了一次 Reward Model vs User Feedback vs Hybrid 对比：
Reward Model：
和 V1 一样，用 RM 对生成会话打 P-score；
用 GBPO 做 RL。
User Feedback Signals：
完全不用 RM；
用 duration-aware reward shaping + GBPO；
样本既可以来自传统系统，也可以来自 OneRec 自己生成的曝光。
Hybrid：
一部分样本来自 RM rollout（reward= P-score）；
一部分样本来自真实反馈（reward = A=±1）；
两类样本都用 GBPO 统一更新。
实验结论是：
两条路都有效：
RM偏向提升交互指标（Like / Comment / Follow）；
用户反馈 reward 偏向提升 App 停留时长。
Hybrid 在多指标平衡上最好




