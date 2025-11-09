# 记录一下用过的多模态模型

## blip2

blip2本质也是个adapter，来对齐视觉backbone和llm decoder。它是个encoder-decoder类型的188M bert，这个adapter叫 **q-former**，关键结构是 **32个768维的可学习embeddings**。  
图像输入经过视觉backbone进入作为v被q查，然后叠12层 cross-attention / self-attention / FFN / LN，最后投影到llm的embeeding层/直接输出文本。

---

### 训练过程：

**Stage 1：视觉-语言表征学习（不接 LLM）**

- **ITC（Image-Text Contrastive）对比学习**：就是 CLIP  
- **ITM（Image-Text Matching）匹配判别**：随机组队，训练二分类  
- **ITG（Image-Grounded Text Generation）**：图在文本指导下生文，自回归 loss（next-token-pred）

其中图像输入头的 embedding 来自视觉 backbone，文本 embedding 来自 bert。  
每轮训练三个任务分别做一次，loss 是三者相加。

**预训练数据（与 BLIP 相同配方，总计约 1.29 亿对）**

- COCO、Visual Genome、CC3M、CC12M、SBU 与 LAION400M 采样 1.15 亿  
- 并用 CapFilt 生成/筛选合成 caption（每图保留 top-2，训练时随机取其一）

---

**Stage 2：视觉 → 语言的生成式对齐（接入 LLM）**

把视觉 backbone 的输出喂给 q-former，得到的 embedding 输出喂给 llm。

**BLIP-2 的 Table 3/4 里面列的视觉 backbone 选项：**

- CLIP ViT-L/14  
- CLIP ViT-g  
- EVA-02-CLIP-Enormous  

---

## qwen3-vl

系列有多个，模型大小 / 是否 MoE / Thinking or Instruct。

视觉 backbone 最早是 openclip，qwen2.5 开始是自己做的 ViT，这次推测也是自己做的，因为 ViT 的隐层维度 hidden_dim 是 **1152**，不是主流那几个 openclip 的；  
语言 backbone 是 qwen3，是否 MoE / Thinking or Instruct 完全是 LLM 的配置。

例如 **Qwen3-VL-235B-A22B-Instruct** 组成如下：

| 模块 | 参数量 |
|:--|--:|
| 视觉编码器 (ViT) | 576.39 M |
| Merger 模块 (总和) | 160.50 M |
| 语言模型 (不含 lm_head) | 233.85 B |
| 语言模型输出层 (lm_head) | 622.33 M |
| **总和** | **235.09 B** |

---

### 模型结构

- `VITconfig.num_position_embeddings = 2304`  
  → 对应 `self.num_grid_per_side = 48`
- `config.deepstack_visual_indexes = [8,16,24]`  
  → 对于一个 27 层的 ViT 来说是比较均匀的。

---

### 关键技术

1. **DeepStack**  
   对 LLM 的第 0、1、2 层，在正常的 forward 结束以后，把 visual embeds 的第 8、16、24 层的特征再重新加回去。

2. **MRoPE-Interleave**  
   原来 “MRoPE” 里，三维视觉 token 的位置 (t, h, w) → 映射到旋转位置编码的不同维度里，是“分段式”；  
   而 Qwen3-VL 改成了 **交错式（interleave）**。  
   这样时间不再只占用最高频维，而是把三个轴均匀打散到所有频率 band。  
   数学上是完全等价的，懂一点表示论能造出一大堆这样的 rope，参见 [这里](https://github.com/happyyzy/position-encoding-and-representation-theory)。

---

### 训练流程

（参考 qwen2.5-vl）

| 训练阶段 | 多模态 token |
|:--|--:|
| Stage-A (Vision pretrain) | ~1.5T token |
| Stage-B (Multimodal joint) | ~2.0T token |
| Stage-C (Long context) | ~0.6T token |

---

#### 阶段 A：视觉预训练（ViT 单独训练）

- window attention 和全局 attention 比例 28:4  
- hidden_dim = 1280，大概 500M  
- 不论 llm 大小用的是这一配置（但不是一套权重）  
- 训练时不像 openclip 直接对比损失，而是冻结 llm 直接训练 vit 和 merger。  
- 使用 2D Rope（回忆一下 nD 情况）

---

#### 阶段 B：多模态预训练（ViT + LLM 全参联合）

- **MRoPE（多模态旋转位置编码）对齐“绝对时间”**：把时间 ID 与绝对时间对齐，而不是仅与“帧数”绑定，提升跨不同 FPS 的时序对齐与事件定位能力。  
- MRoPE 实质：视频用 3D rope，图像用 2D rope 等；创新点在于“绝对时间”。
- **3D patch / 成对帧分组**：两帧成组、降 token 量，进一步控制视频场景的序列长度与算力。

---

#### 阶段 C：长上下文预训练（ViT + LLM 联合，放大序列）

- 序列长度：前两阶段 8192，长上下文阶段 32768。  
- 继续联合训练，但把序列长度扩到 32,768。

## clip
两个输入头，这点和blip2不一样（ert初始化直接能得到语义向量），有钱有数据集，超大batch（提升对比度）喂给infonce直接训练就行了。
注意它的结构：它有一个初始token[cls],代表全局图像/语义，而不是单纯的patch embedding做自注意力。这是2020vit论文就提出的。

## controlnet
对unet每个block在旁边复制一个control branch,branch最后接一个初始为0的1×1conv来保证训练开始时输出一致，每个block的输出是unet输出加上branch的输出，
注意这里复制的branch权重初始和unet一样非零，保证了初始梯度不为0，类似lora的初始化。

## ip-adapter
Tencent ARC Lab（吕昊翔团队）在 2023 年提出。
就是找个vit，比较细致的方案会用patch向量，加一些投影层和resampler(可学习的q,多层注意力来固定个数和维度)接在文本kv的后面。

## dinov2-large
[gpt聊天](https://chatgpt.com/share/690f28a6-5fa0-8010-90e4-9b7c06a27cb2)
先按照时间顺序讲dino,ibot,sk算法
### prototype自蒸馏
DINO（2021）提出无标签自蒸馏：student 去拟合由 EMA teacher 的[cls]给出的softmax 分布（prototype），并用 centering + sharpening 防塌缩，但并未采用原型均衡（SK）。

### Sinkhorn–Knopp算法
# Sinkhorn–Knopp（1967–1969，Richard Sinkhorn & Paul Knopp）

**核心问题：矩阵平衡（matrix scaling）**

给定一个非负矩阵 $A$，是否存在两个对角缩放矩阵

- $D_r = \mathrm{diag}(u)$  
- $D_c = \mathrm{diag}(v)$

使得

$$
B \;=\; D_r\,A\,D_c
$$

的**每一行**与**每一列**都满足指定的边缘分布（row-sums / column-sums）？

> 典型应用：构造双随机矩阵（doubly-stochastic matrix）、OT / Sinkhorn divergence 等等。

---

## 迭代更新（最核心的两行）

给定当前 $v$，更新 $u$：

$$
u_k
=
\frac{B/K}{\displaystyle \sum_{b} P_{kb}\,v_b}
$$

给定当前 $u$，更新 $v$：

$$
v_b
=
\frac{1}{\displaystyle \sum_{k} u_k\,P_{kb}}
$$

---

一句话总结：

> **反复归一化行、列 → 逼近满足指定边缘分布的缩放矩阵。**


### iBOT
iBOT（Image BERT Pre-Training with Online Tokenizer）
——是 2021 年由百度视觉团队提出的自监督视觉预训练框架（论文发表于 CVPR 2022）。它直接承接了 DINO 的 “teacher–student 自蒸馏” 思路，同时引入了 BERT-式 mask-预测机制。
# iBOT 概要笔记

## 一、总体思路

**iBOT 的核心目标：**

> **让 student 网络在部分 patch 被 mask 的情况下，仍能预测出这些 patch 在 teacher 特征空间中的语义表示。**

- **teacher**：结构与 student 相同，用 **EMA** （exponential moving average）更新参数，**看到完整图像**
- **student**：输入图像随机 mask 掉部分 patch，仅凭剩余上下文**预测被 mask patch 的语义**
- **tokenizer**：teacher 输出的 patch feature → projection → softmax → 与有限个 **“prototype” codebook** 对应  
  → 类似 BERT 的 masked token prediction  
  但预测空间不是词表 ID，而是 **teacher 的 prototype 语义编码**

## 二、网络结构与目标

- 骨干：**ViT**（Vision Transformer），输入为固定 patch
- 两个 branch：

| branch | 目标 |
|--------|------|
| Global（全局视角） | 沿袭 DINO，对 `[CLS]` token 做整体语义蒸馏 |
| Local（局部视角）  | 新增 patch-level 蒸馏，对每个 patch token 做预测 |

- mask 策略：**随机屏蔽 40%–75%** patch，仅给 student 剩余 patch

---

### teacher 输出

$$
z_t = \text{softmax}\left(\frac{f_t(x)}{\tau_t}\right)
$$

> $\tau_t$ = teacher 温度

### student 输出

$$
z_s = \text{softmax}\left(\frac{f_s(\mathrm{Mask}(x))}{\tau_s}\right)
$$

### patch-level loss

$$
\mathcal{L}_{\text{iBOT}}
= - \sum_{i \in \text{masked}} 
\, z_{t,i} \cdot \log z_{s,i}
$$

> 在 mask 位置对齐 teacher 与 student 的分布

### teacher 参数更新（EMA）

$$
\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s
$$

典型动量：  
$$
m \approx 0.996
$$

---

## 三、训练细节

- **数据增强**：multi-crop + 颜色扰动 + 高概率 mask
- **温度**：teacher $\tau = 0.04$，student $\tau = 0.1$
- **prototype 数**：**8192**
- **优化器**：AdamW；weight decay **0.04 → 0.4**（cosine schedule）
- **teacher 动量**：**0.996 → 1.0**（cosine schedule）
- **loss**：global+local两部分（cosine schedule）

最后回到dinov2，有以下几个改进
1.dino loss计算：teacher 端：softmax 后再做 居中（centering），推荐用 Sinkhorn–Knopp（SK）3 次迭代
2.iBOT 共享投影头、DINOv2 取消共享


## deepface

## LongCat-Video



