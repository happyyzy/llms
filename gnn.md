# 推荐算法相关的gnn
arXiv:1403.6652是随机游走embedding的开山之作，目前引用量已经过万，解释了deepwalk，node2vec本质上是做矩阵分解，和nlp早期的embeeding方案/推荐的协同过滤跟出同源。
# deepwalk
## 定理 1（随机游走出现频率极限）

当随机游走长度 L→∞,∣D→r​∣#(w,c)→r​​→vol(G)dw​​(Pr)w,c​
同理反向窗口(c→w) 亦然。

文中用到的技术：
S.N. Bernstein LLN（弱相关序列的强律 + random walk 收敛到 stationary）
→ 用于将“样本计数” → “转移概率” 极限。
其实完全没必要，熟悉markov chain就是废话，因为有限图positively reccurent，连通图irriducible,非二部图iperiodic，所以无限长随机游走必定收敛到稳定分布。
论文相当于在此特殊情况下又证明了一遍收敛定理，可能是多来点术语撑排面用吧。。

## 定理2 
对定理1求和，得到长度T的窗口出现pair的概率∣D∣#(w,c)​→2T1​r=1∑T​[vol(G)dw​​(Pr)w,c​+vol(G)dc​​(Pr)c,w​]
另外稳定分布告诉我们∣D∣#(w)​→vol(G)dw​​

## 定理3 (Omer Levy, Yoav Goldberg. Neural word embedding as implicit matrix factorization. NIPS 2014.)
SGNS 是在 factorize shifted PMI log(PMI fraction)−logb。
此理论分析有两个假设：
1.embedding 维度高=每个 dot product 独立
2.负采样分布PN是图的稳定分布，而不是实际常用的3/4次方
所以我们得到结论:DeepWalk factorize 的 closed–form是log(vol(G)T1​r=1∑T​PrD−1)−logb

现在我们可以解释item2vec为什么更适用于稀疏场景，提升低频item的表达能力了：因为vol(G)T1​r=1∑T​PrD=D−1/2U(T1​r=1∑T​Λr)U⊤D−1/2是一种典型的低通滤波器，
保留了item图的低频振动形式，使得embedding分布非常平滑。
这个filter其实在谱聚类就有类似思想：归一化的邻接矩阵对多次求幂谱稳定，进而出现在GCN里面作为第一项，其实就是xr这个滤波器。
GCN第二项可以理解为一种残差链接,或者是一种数值稳定trick：filter变成(x+epsilon)r，使得-1到1的频段往正方向平移了一点，在做高次幂的时候更好过滤高频段和保留低频段。
另外这个根号的归一化其实和attention归处同源。
另外类似的思想在数学里也很常见，f(x)=T1​∑xr这个filter把Hilbert space的operater过滤成（强算子拓扑收敛）其不动点空间的正交补上（graph情形就是归一化拉普拉斯的kernel，也即低频特征），
这就是opertor版本的von Neumann 平均遍历定理，套到Koopman算子即可证明L2版本的遍历定理。我是在rudin的泛函分析看到这个证明，和传统的证明完全不同，深觉有趣。

# node2vec
分析过程几乎一模一样，只不过此时是二阶markov chain。直接给出定理吧：
(#(w)/∣D∣)(#(c)/∣D∣)#(w,c)/∣D∣​--->2T1​r=1∑T​(∑u​Xw,u​∑u​Xw,u​(Pr)c,w,u​​+∑u​Xc,u​∑u​Xc,u​(Pr)w,c,u​​)
其中Xv,w是二阶随机游走的 stationary distribution over ordered pairs

# lightgcn
我们在rec.md里讲了pointwise的deepfm，listwise的lambda,现在刚好有个pairwise的lightgcn。
架构很简单，只有use/item的embbeding,params=(nusers​+nitems​)×d,三层无参数无self-loop的GCN（就是邻接矩阵传播三次），数据集也很简单，没有各种features。估计大小：
Amazon-Book
9,231,488
36.93 MB

loss是BPR+L2正则化，也是一种对比loss，不同之处是这里是对同1个用户各采样一个正负样本
LBPR​=−u∑​i∈Nu​∑​j∈/Nu​∑​lnσ(y^​ui​−y^​uj​)+λ∥E(0)∥2








