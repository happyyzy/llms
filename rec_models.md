# ROC曲线
指标	宾语集合	问的问题
Recall (TPR)	正例 P	抓得全吗？（漏没漏）
FPR	负例 N	冤枉了多少？（错抓）
横轴fpr，纵轴tpr

阈值极高(T=1)	阈值极低(T=0)
模型特别严 → 很难判正	模型特别松 → 几乎全判正
正例和负例都很难判成Positive	正例负例都很容易判成Positive
TPR、FPR 都接近 0	TPR、FPR 都接近 1

阈值	点
T = +∞ 或非常接近 1	(FPR=0, TPR=0)
T = 0	(FPR=1, TPR=1)
这样就得到roc曲线，围的面积auc就代表排序水平。auc=1就是完美分类。

# 模型结构
全部 item（可能 1千万）
      │
      ▼
召回层 Recall（双塔）
      ↓ 选出几百～几千候选
排序层 Ranking（更复杂模型）
      ↓ 选出几十个展示
展示

召回层的双塔 就和clip一样，训练loss也一样,对不同的特征用相应的流程得到相应的embbeding再pool到一起，比如

视频推荐：item塔可能是 “视频封面 CNN + 文本BERT + 分类embedding + MLP 融合”

电商商品：title 用 transformer 编成 embedding

给出新用户，用召回模型选出候选item，现在要做排序。

排序时预测目标CTR，就像经典树模型一样，要做特征交互，我们看一下三种
Wide deep FM
Wide 加入手工交叉特征，线性模型升级版，能捕捉明确的、强关联的组合行为，记忆力强，无法泛化没见过的组合
Deep 学习高阶特征交互，泛化能力强，不需要过多人工设计特征，但是容易“过度泛化”，忽略一些重要的、稀疏的组合，稀疏的、强规则性的特征组合不准确
FM, FM 是一种自动学习二阶特征交互的模型，特别适合稀疏数据

# deepfm
# FM / DeepFM —— 离散 / 连续特征的统一表示

## 0）统一符号

一条样本包含若干 field：

$$
\{ f_1, f_2, \ldots, f_m \}
$$

embedding 维度：$d$

线性项权重：第 $i$ 个 field 的线性系数：

$$
w_i
$$

FM “二阶交互”使用 embedding：

$$
e_i \in \mathbb{R}^d
$$

---

## 1）离散特征（Categorical）

### 1.1 含义

- user_id / item_id / item_cate / gender / weekday / …
- 每个类别学习一个 embedding

### 1.2 主流处理

- 先编码成 id  
- 每个 field 建 embedding matrix：

$$
E_i \in \mathbb{R}^{(\text{num\_classes}_i)\times d}
$$

样本取值 $c$ → lookup 得：

$$
e_i = E_i[c]
$$

multi-hot：先在 field 内求 “sum / mean / weighted mean”

线性项：

$$
linear_i = w_{i,c}  \quad (\text{或 multi-hot 的加权和})
$$

### 1.3 FM / Deep

FM 二阶：

$$
y_{FM}^{(2)} = \sum_{i<j} \langle e_i , e_j \rangle
$$

Deep 输入：

$$
z = concat(e_1,e_2,\ldots,e_m)\in\mathbb{R}^{m\cdot d}
$$

---

## 2）连续特征（Numeric / Real）

### 2.1 含义

- price / age / duration / rating / timestamp …

### 2.2 两条路线

#### 路线 A：标准化 + scalar-scaled embedding（DLRM / DeepFM 常用）

标准化：

$$
x_i' = \frac{x_i-\mu_i}{\sigma_i}
\quad \text{或} \quad
x_i' = \frac{x_i - \min}{\max-\min}
$$

线性项：

$$
linear_i = w_i \cdot x_i'
$$

field embedding：

$$
e_i = v_i \cdot x_i' ,\quad v_i\in\mathbb{R}^d
$$

#### 路线 B：分桶 → 当成离散 embedding

切 $K$ 桶 → bucket id：

$$
e_i = E_i[bucket(x_i)]
$$

> 工业常 A+B 混用（数值通道 + 分桶通道）

### 2.3 FM / Deep（沿用统一公式）

FM：

$$
y_{FM}^{(2)} = \sum_{i<j} \langle e_i , e_j \rangle
$$

Deep：

$$
z = concat(e_1,\ldots,e_m),\quad
y_{Deep}=MLP(z)
$$

---

## 3）总体输出（DeepFM）

FM：

$$
y_{FM} = w_0 + \sum_i linear_i + \sum_{i<j} \langle e_i,e_j\rangle
$$

Deep：

$$
y_{Deep} = MLP(concat(e_1,\ldots,e_m))
$$

最终（CTR）：

$$
\hat{y} = \sigma(y_{FM}+y_{Deep})
$$

---

## 4）实务要点

- loss就是预测ctr和实际是否点击的交叉熵，可能会加权
- 连续特征必须标准化（线上同分布）
- 缺失值要有 default + 缺失指示器
- multi-hot 推荐加权 sum / mean / attention 汇聚
- 离散长尾要截断
- A+B 混用通常更稳
- 数值尺度影响 $e_i = v_i\cdot x_i'$ ——要监控激活分布

  # 一点item2vec
原始idea:从文本里面划window，得到正负样本，优化点积的softmax
后续优化：做随机游走来取window，做负采样，用sigmoid代替softmax，这里也是一种对比学习。
后续衍生：GNN的node2vec,推荐的item2vec等等。
为什么要用用sigmoid代替softmax？首先实验上有明显优势，其次理论上sigmoid对应PMI矩阵分解，能像svd一样写出显示最优解，我们在这里总结了各种embedding和矩阵分解的关系。
DeepWalk 更适合稀疏场景，能提升低频 item 的表示能力。事实上确实是一种低通滤波。

上面的deepfm是pointwise,下面介绍一个更强力的listwise方法：Lambda 系列。
# LambdaLoss
NDCG作为目标函数衡量list排序的正确性，它对模型输出score不可导，于是引入λij​=ΔNDCGij​⋅∂(si​−sj​)∂Lij​​当作对score的梯度，
λi​=j:(i,j)∑​λij​−j:(j,i)∑​λji​作为回传的梯度。具体来说：
两种情况：NN vs LambdaMART
① 如果你的 ranker 是神经网络

那流程就是：

1）前向计算得到 每个 item 的 score(s_i)
2）根据 score 两两比较得到 λ_i（对每个样本的梯度）
3）把 λ_i 视为 dL/ds_i
4）再往下 backprop，把梯度传给网络参数 θ：
θ←θ−η∂θ∂s​⋅λ
也就是说：

Lambda 只是“替换了 dLoss/dScore”，其它全是标准 BP。

换成一句话就是：

Lambda = gradient on score
→ backprop to parameters

② 如果你是 LambdaMART（GBDT）

Tree 没有“参数”矩阵，所以更新方式是 boosting：

1）对每个样本有 λ_i
→ 这代表 “当前模型对 s_i 的残差”

2）训练一棵树去拟合 λ_i

3）把这棵树加入ensemble：
Ft+1​(x)=Ft​(x)+η⋅treet​(x)

所以 LambdaMART：

Lambda 用来做 “残差目标值”，树拟合它
再把树累加上去

没有 SGD
是 boosting 叠树
# 冷启动
两种冷启动，怎么对齐embedding格式

# 业务指标
CTR (Click-Through Rate)：点击率
CVR (Conversion Rate)：转化率

# 实际在线排序过程

