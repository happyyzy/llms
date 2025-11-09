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

召回层的双塔 就和clip一样，训练loss也一样

给出新用户，用召回模型选出候选item，现在要做排序。

# deepfm
每个类别特征都有embbeding
