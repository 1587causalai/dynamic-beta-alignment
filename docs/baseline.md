# Beta-DPO基准测试指南

## 环境配置

1. 克隆原始仓库
```bash
git clone https://github.com/junkangwu/beta-DPO.git
cd beta-DPO
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 数据准备

Beta-DPO支持以下数据集:
- Anthropic-HH
- Stanford-HH
- OpenAI-summarize-from-feedback
- Dahoas/rm-static

数据集会在首次运行时自动下载到本地。

## 运行基准测试

1. 基本训练命令
```bash
python train.py --config config/config.yaml
```

2. 常用配置选项
- 修改模型: 更改`config/config.yaml`中的`model_name_or_path`
- 调整β值: 更改`config/loss/dpo.yaml`中的`beta`参数
- 数据集选择: 更改`config/config.yaml`中的`dataset_name`

3. 典型实验配置

a) GPT2-Large + Anthropic-HH (原论文主要设置)
```yaml
# config/config.yaml
model_name_or_path: gpt2-large
dataset_name: Anthropic/hh-rlhf
beta: 0.1  # 论文推荐值
```

b) LLaMA-7B + Stanford-HH
```yaml
# config/config.yaml
model_name_or_path: decapoda-research/llama-7b-hf
dataset_name: stanfordnlp/SHP
beta: 0.1
```

## 评估指标

主要评估指标包括:
1. Preference Accuracy: 模型对齐人类偏好的准确率
2. Reference KL: 与参考模型的KL散度
3. Win Rate: 与基线模型的胜率对比

## 实验记录模板

```markdown
### 实验配置
- 模型: [模型名称]
- 数据集: [数据集名称]
- β值: [beta值]
- 其他参数: [列出重要参数]

### 实验结果
- Preference Accuracy: [数值]
- Reference KL: [数值]
- Win Rate: [数值]
- 训练时长: [时间]
- GPU使用: [显存使用情况]

### 观察结论
[记录实验现象和结论]
```

## 常见问题

1. 内存不足
- 减小batch_size
- 使用gradient_accumulation_steps
- 启用gradient_checkpointing

2. 训练不稳定
- 调整learning_rate
- 修改beta值
- 检查数据预处理

## 实验计划

1. 复现论文主要结果
   - [ ] GPT2-Large on Anthropic-HH
   - [ ] LLaMA-7B on Stanford-HH

2. 额外实验
   - [ ] 不同β值的对比实验
   - [ ] 不同数据集的泛化性实验
   - [ ] 模型规模的影响实验

## 注意事项

1. 资源需求
   - GPT2-Large: 最少16GB显存
   - LLaMA-7B: 最少32GB显存

2. 实验建议
   - 建议先用小模型快速迭代
   - 保存关键checkpoint
   - 详细记录实验配置和结果 