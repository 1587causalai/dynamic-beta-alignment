# DPO 损失函数推导

## 1. 问题设定

在强化学习框架下，我们的目标是找到一个最优策略 $\pi(y|x)$，使得期望奖励最大化。同时，我们希望新策略不要偏离参考策略 $\pi_{\text{ref}}(y|x)$ 太远。这可以形式化为如下优化问题：

$$
\max_{\pi} \mathbb{E}_{x \sim d, y \sim \pi(y|x)}[r(x, y)] \quad \text{s.t.} \quad \mathbb{E}_{x \sim d}[D_{KL}(\pi(\cdot|x) || \pi_{\text{ref}}(\cdot|x))] \leq \epsilon
$$

其中：
- $r(x, y)$ 是奖励函数
- $d$ 是输入分布
- $D_{KL}$ 是 KL 散度
- $\epsilon$ 是允许的最大 KL 散度

## 2. 拉格朗日乘子法

使用拉格朗日乘子法将约束优化问题转化为无约束优化问题：

$$
\mathcal{L}(\pi, \beta) = \mathbb{E}_{x \sim d, y \sim \pi(y|x)}[r(x, y)] - \beta(\mathbb{E}_{x \sim d}[D_{KL}(\pi(\cdot|x) || \pi_{\text{ref}}(\cdot|x))] - \epsilon)
$$

其中 $\beta \geq 0$ 是拉格朗日乘子。

## 3. 最优策略形式

### 3.1 变分推导

对于固定的 $\beta$，最优策略的形式可以通过变分法得到。考虑策略 $\pi$ 的微小变分 $\delta \pi$，在最优点处应有：

$$
\left.\frac{d}{d\epsilon}\mathcal{L}(\pi + \epsilon\delta\pi, \beta)\right|_{\epsilon=0} = 0
$$

这导致：

$$
\pi^*(y|x) = \pi_{\text{ref}}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)/Z(x)
$$

其中 $Z(x)$ 是归一化因子。

### 3.2 直接参数化

在实践中，我们直接参数化策略 $\pi_\theta(y|x)$，并最小化如下目标：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{x \sim d, y \sim \pi_\theta(y|x)}[r(x, y)] + \beta D_{KL}(\pi_\theta(\cdot|x) || \pi_{\text{ref}}(\cdot|x))
$$

## 4. 偏好学习设定

在偏好学习中，我们通常有成对的比较数据 $(x, y_w, y_l)$，其中 $y_w$ 是偏好选择，$y_l$ 是非偏好选择。这种情况下：

### 4.1 奖励建模

我们可以定义隐式奖励：

$$
r(x, y_w) - r(x, y_l) = \log\frac{\pi^*(y_w|x)}{\pi^*(y_l|x)} = \log\frac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)} + \frac{r(x,y_w) - r(x,y_l)}{\beta}
$$

### 4.2 二元交叉熵损失

最终的 DPO 损失可以简化为二元交叉熵形式：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\frac{1}{\beta}\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \frac{1}{\beta}\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 是 sigmoid 函数。

## 5. 动态化 $\beta(x)$ 的推广

在我们的改进方案中，将固定的 $\beta$ 替换为依赖于输入 $x$ 的函数 $\beta(x)$：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\frac{1}{\beta(x)}\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \frac{1}{\beta(x)}\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

这种推广允许模型根据输入动态调整奖励信息和参考策略的权重，提供了更大的灵活性。

### 5.1 正则化考虑

为了确保 $\beta(x)$ 的合理性，我们添加以下正则化项：

1. **范围约束**：
   $$\mathcal{L}_{\text{range}} = \lambda_1\mathbb{E}_x[\max(0, \beta(x) - \beta_{\text{max}}) + \max(0, \beta_{\text{min}} - \beta(x))]$$
   
   这个约束确保 $\beta(x)$ 的值始终在合理范围内，防止过大或过小。

2. **平滑性约束**：
   $$\mathcal{L}_{\text{smooth}} = \lambda_2\mathbb{E}_x[||\nabla_x\beta(x)||^2]$$
   
   这个约束需要从任务/场景一致性的角度重新思考：
   
   **基本原则**：
   - $\beta(x)$ 应该主要反映任务类型而不是具体输入的特征
   - 同类型任务（如数学问题）应该有相似的 $\beta(x)$ 值
   - 不同类型任务之间 $\beta(x)$ 可以有较大差异
   
   **场景示例**：
   - **数学问题**：较小的 $\beta(x)$
     - 奖励信号（答案正确性）非常可靠
     - 可以更大胆地偏离参考策略
     - 在数学题内部，$\beta(x)$ 应该相对稳定
   
   - **开放对话**：较大的 $\beta(x)$
     - 奖励信号（对话质量评分）较为主观
     - 需要更多地保持接近参考策略
     - 在对话任务内部，$\beta(x)$ 也应该相对稳定
   
   - **代码生成**：中等的 $\beta(x)$
     - 奖励信号（功能正确性）相对可靠
     - 但实现方式可以多样
     - 在相似类型的编程任务中，$\beta(x)$ 应该接近
   


3. **熵相关性约束**：
   $$\mathcal{L}_{\text{entropy}} = \lambda_3\mathbb{E}_x[(\beta(x) - \alpha H(\pi_{\text{ref}}(\cdot|x)))^2]$$
   
   这个约束基于以下直觉：
   - $H(\pi_{\text{ref}}(\cdot|x))$ 是参考策略在输入 $x$ 下的熵，表示模型的不确定性
   - 当参考策略的熵较高时（即模型不确定性大），我们希望 $\beta(x)$ 也相应增大，更多地依赖参考策略
   - 当参考策略的熵较低时（即模型比较确定），我们可以使用较小的 $\beta(x)$，更多地依赖奖励信号
   - $\alpha$ 是一个缩放因子，用于调整熵和 $\beta(x)$ 的比例关系

   举例来说：
   - 对于一个简单、明确的输入（如 "1+1="），参考策略的熵会很低，此时可以更多地信任奖励信号
   - 对于一个复杂、模糊的输入（如 "讲个笑话"），参考策略的熵会很高，此时应该更多地保持接近参考策略

最终的损失函数为：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{range}} + \mathcal{L}_{\text{smooth}} + \mathcal{L}_{\text{entropy}}
$$

其中各个 $\lambda_i$ 是权衡不同正则化项重要性的超参数。

## 6. 实现考虑

在实践中，我们需要注意以下几点：

1. **数值稳定性**：使用 log-space 计算避免数值溢出
2. **梯度裁剪**：防止梯度爆炸
3. **批量归一化**：在 $\beta(x)$ 网络中使用批量归一化
4. **预训练初始化**：使用固定 $\beta$ 的预训练结果初始化 