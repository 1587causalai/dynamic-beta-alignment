# 预备知识：从 Transformer 到 SFT

## 1. Transformer 架构基础

### 1.1 整体架构

Transformer 采用编码器-解码器架构，核心组件包括：
- 多头自注意力机制（Multi-head Self-attention）
- 前馈神经网络（Feed-forward Network）
- 层归一化（Layer Normalization）
- 残差连接（Residual Connection）

<div style="display: flex; justify-content: space-between;">
    <img src="https://s2.loli.net/2024/12/25/clFVeJPvOkg958W.png" width="35%" alt="Transformer Architecture">
    <img src="https://s2.loli.net/2024/12/25/19lvtwpCH5akIYW.png" width="63%" alt="Transformer Details">
</div>

### 1.2 注意力机制

#### 1.2.1 一般注意力机制
注意力机制最初用于序列到序列的转换任务（如机器翻译），其中：
- Query (查询) 来自目标序列
- Key (键) 和 Value (值) 来自源序列
- 通过计算 Query 和 Key 的相似度，得到注意力权重
- 用这些权重对 Value 进行加权求和

为了保持层与层之间的连接性，注意力机制需要维持输入输出维度的一致性：
- 输入序列：$I \in \mathbb{R}^{n \times d_{model}}$
- 输出序列：$O \in \mathbb{R}^{n \times d_{model}}$

例如在情感分析任务中：
- 输入："这部电影很好看，但是结局有点仓促"
- Query 可能是我们要分类的目标特征
- Key 和 Value 是句子中的词语表示
- 注意力机制会自动关注到"好看"、"仓促"等情感词

#### 1.2.2 缩放点积注意力
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中，$I$ 是输入序列，$W^Q$, $W^K$, $W^V$ 是可训练的参数矩阵：
- $Q = IW^Q \in \mathbb{R}^{n \times d_k}$ 是查询矩阵
- $K = IW^K \in \mathbb{R}^{m \times d_k}$ 是键矩阵
- $V = IW^V \in \mathbb{R}^{m \times d_v}$ 是值矩阵，其中 $d_v = d_k = d_{model}$
- $\sqrt{d_k}$ 是缩放因子，防止点积结果过大导致 softmax 梯度消失

维度分析：
1. $QK^T$: $(n \times d_k)(m \times d_k)^T = n \times m$
2. $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})$: $n \times m$
3. 最终输出: $(n \times m)(m \times d_v) = n \times d_v$，其中 $d_v = d_{model}$

#### 1.2.3 自注意力机制 (Self-Attention)
自注意力是注意力机制的特殊情况，其中：
- Query、Key 和 Value 都来自同一个序列（即 $n = m$）
- 必须保持 $d_k = d_v = d_{model}$，因为：
  1. 输入序列维度是 $d_{model}$
  2. 输出序列维度也必须是 $d_{model}$（为了堆叠多层）
  3. 这确保了层与层之间可以直接连接

![20241225100620](https://s2.loli.net/2024/12/25/TnWYEupgRSVb2kX.png)

#### 1.2.4 多头自注意力 (Multi-Head Self-Attention)
为了增强模型的表达能力，将自注意力机制扩展为多个"头"：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

其中：
1. 每个头独立计算注意力：
   $$
   \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
   $$

2. 维度关系：
   - 输入维度：$d_{model}$
   - 每个头的维度：$d_k = d_v = d_{model}/h$
   - 每个头的参数矩阵：$W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{model} \times (d_{model}/h)}$
   - 拼接后的维度：$h \times (d_{model}/h) = d_{model}$
   - 输出投影：$W^O \in \mathbb{R}^{d_{model} \times d_{model}}$
   - 最终输出维度：$d_{model}$（与输入保持一致）


![20241225104452](https://s2.loli.net/2024/12/25/owsK1d7D3a4NPjk.png)

3. 优点：
   - 每个头可以关注不同的特征模式
   - 增加了模型的并行性
   - 提供了多个表示子空间

**注意力机制的关键优势**：
1. **处理变长序列**：
   - 传统 RNN 需要固定的隐藏状态大小，处理长序列时信息容易丢失
   - 注意力机制通过 $QK^T$ 计算相关性，可以直接处理任意长度的输入
   - 参数矩阵与序列长度无关，只与隐藏维度相关

2. **并行计算**：
   - 所有位置的注意力权重可以同时计算
   - 不像 RNN 需要序列化处理

### 1.3 位置编码

使用正弦和余弦函数的位置编码：
$$
\begin{aligned}
PE_{(pos,2i)} &= \sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)} &= \cos(pos/10000^{2i/d_{model}})
\end{aligned}
$$

### 1.4 前馈网络

两层线性变换与一个非线性激活函数：
$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

## 2. 语言模型训练

### 2.1 自回归语言模型

给定上文预测下一个词的概率：
$$
p(x_t|x_{<t}) = \text{softmax}(h_t W + b)
$$

其中 $h_t$ 是最后一层的隐藏状态。

### 2.2 负对数似然损失

语言模型训练的目标是最大化数据的似然概率。对数似然为：
$$
\log p(x) = \sum_{t=1}^T \log p(x_t|x_{<t})
$$

最小化负对数似然（Negative Log-Likelihood, NLL）：
$$
\mathcal{L}_{\text{NLL}} = -\sum_{t=1}^T \log p(x_t|x_{<t})
$$

这个损失函数：
1. 直接来自于最大似然估计（MLE）原理
2. 在分类问题中，与交叉熵损失在形式上等价
3. 但推导过程和理论基础不同：
   - 交叉熵基于信息论
   - NLL 基于统计学的最大似然估计

## 3. SFT（Supervised Fine-Tuning）

### 3.1 基本原理

SFT 是在预训练语言模型基础上，使用高质量的指令-回复数据进行微调的过程。

### 3.2 数据格式

典型的 SFT 数据包含：
- 指令/提示（Prompt）：$x$
- 目标回复（Response）：$y$

### 3.3 训练目标

SFT 的训练目标是最大化条件概率：
$$
\mathcal{L}_{\text{SFT}} = -\sum_{(x,y)\in D} \log p_\theta(y|x)
$$

展开到 token 级别：
$$
\mathcal{L}_{\text{SFT}} = -\sum_{(x,y)\in D} \sum_{t=1}^{|y|} \log p_\theta(y_t|y_{<t}, x)
$$

### 3.4 实现细节

1. **提示工程**：
   - 统一的提示格式（如 "Human: ... Assistant: ..."）
   - 特殊标记的使用（如 [SOS], [EOS]）

2. **注意力掩码**：
   - 确保模型不能看到未来的 tokens
   - 实现单向注意力机制

3. **训练技巧**：
   - 学习率预热（Warmup）
   - 梯度裁剪
   - 混合精度训练

### 3.5 评估指标

1. **困惑度**（Perplexity）：
   $$
   \text{PPL} = \exp(\mathcal{L}_{\text{NLL}})
   $$
   
   通俗理解：
   - PPL 表示模型在每个位置平均需要从多少个词中进行选择
   - PPL 越小，说明模型的预测越自信，越确定下一个词应该是什么
   - 例如：PPL = 10 意味着模型在每个位置平均在 10 个可能的词中犹豫
   - 完美模型的 PPL = 1，表示每次都能准确预测下一个词

2. **生成质量指标**：
   - BLEU（机器翻译）
   - ROUGE（摘要生成）
   - 人工评估分数

## 4. 从 SFT 到 DPO

### 4.1 SFT 的局限性

1. **反馈信号有限**：
   - 只能学习单一的"正确"回答
   - 难以捕捉细微的质量差异

2. **优化目标单一**：
   - 仅优化似然度
   - 缺乏对输出质量的直接优化

### 4.2 为什么需要 DPO

1. **更丰富的反馈信号**：
   - 利用偏好数据
   - 可以学习细微的质量差异

2. **更直接的优化目标**：
   - 直接优化输出质量
   - 保持与基础模型的适当距离

### 4.3 衔接点

SFT 通常作为 DPO 的基础模型：
1. 先进行 SFT 得到一个基本对齐的模型
2. 再使用 DPO 进行更细致的优化

## 5. 实践注意事项

1. **数据质量控制**：
   - 数据清洗和过滤
   - 格式统一化
   - 质量评估

2. **训练稳定性**：
   - 合适的学习率选择
   - 梯度累积
   - 检查点保存

3. **评估和监控**：
   - 训练损失曲线
   - 验证集性能
   - 生成样本质量 