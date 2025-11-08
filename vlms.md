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







