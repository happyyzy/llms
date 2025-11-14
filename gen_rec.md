# 生成式推荐模型小结

# one-rec V1
端到端生成，encoder-decoder结构。生成格式：推荐的list
如何把输出头的推荐离散化：连续的embbeding划网格，找最近格点；再划细分网格，找最近格点。。。以此类推就得到格点序列，成功把连续输出化成离散的item ids。
用了MOE.
参数规模
OneRec-0.015B、OneRec-0.121B、OneRec-0.935B、OneRec-2.633B四个版本
很多扩展/消融是 以 0.935B 为基准做的（例如训练收敛样本量、后续若干 scaling 与消融实验）
训练数据量
论文写得很清楚：
2.3B user sequences
（是 23亿会话序列这个级别）
训练分预训练+RL，没有传统llm的sft。
训练时间
论文给的是 “数周”＋“多轮 IPA（DPO）”
因为 IPA 是 iterative preference alignment，它本身是多轮 RL / DPO 微调，论文也说明每一轮只抽 1% 样本做 DPO。



