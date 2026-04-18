# RLTrainPPT - 基于强化学习的PPT内容生成

本项目使用 **GSPO (Group Sequence Policy Optimization)** 强化学习方法训练大语言模型，实现自动生成高质量的演示文稿大纲和内容。

## 📁 项目结构

```
backend/
├── .env                    # 环境变量配置（API密钥、模型配置等）
├── requirements.txt        # Python 依赖包
├── README.md              # 本文档
│
├── outline/               # 大纲生成模块
│   ├── prompt.py          # 大纲生成的提示词
│   ├── train_trl.py       # GSPO 训练脚本
│   ├── model_test.py      # 模型推理测试脚本
│   ├── topic.json         # 训练主题列表
│   ├── outline.jsonl      # 生成的大纲数据
│   └── output/            # 训练输出目录
│
└── content/               # 内容生成模块
    ├── prompt.py          # 内容生成的提示词
    ├── train_trl.py       # GSPO 训练脚本
    ├── model_test.py      # 模型推理测试脚本
    ├── content.jsonl      # 生成的内容数据
    └── output/            # 训练输出目录
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建 conda 环境（推荐 Python 3.10+）
conda create -n rlppt python=3.10
conda activate rlppt

# 安装依赖
cd backend
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `backend/.env` 文件：

```bash
# ===========================================
# 通用配置
# ===========================================

# 基础模型配置
ART_MODEL=Qwen/Qwen2.5-0.5B-Instruct
MAX_SEQ_LEN=4096

# DeepSeek API 配置（用于奖励函数评估）
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-chat
USE_DEEPSEEK_JUDGE=true

# ===========================================
# Outline 训练配置
# ===========================================
OUTLINE_NAME=outline01
OUTLINE_PROJECT=outline-training

# ===========================================
# Content 训练配置
# ===========================================
CONTENT_NAME=content01
CONTENT_PROJECT=content-training
```

## 📝 使用流程

整个项目分为两个阶段：**大纲生成** → **内容生成**

### 阶段一：大纲生成

#### 1.1 训练大纲生成模型

```bash
cd backend/outline
python train_trl.py
```

训练过程会：
- 加载基础模型（默认 Qwen2.5-0.5B-Instruct）
- 使用 `topic.json` 中的主题进行训练
- 通过规则奖励函数 + DeepSeek API 评估进行强化学习
- 模型保存到 `output/outline01-{timestamp}/`

#### 1.2 测试大纲生成

```bash
cd backend/outline
python model_test.py
```

这会：
- 加载训练好的模型
- 为测试主题生成大纲
- 保存结果到 `outline.jsonl`

### 阶段二：内容生成

#### 2.1 训练内容生成模型

```bash
cd backend/content
python train_trl.py
```

训练过程会：
- 加载 `outline/outline.jsonl` 作为训练数据
- 训练模型根据完整大纲生成详细内容
- 模型保存到 `output/content01-{timestamp}/`

#### 2.2 测试内容生成

```bash
cd backend/content
python model_test.py
```

这会：
- 加载训练好的内容生成模型
- 输入完整大纲，生成详细内容
- 保存结果到 `test_results_{timestamp}.jsonl`

## 🔧 核心配置说明

### 训练参数（在 `train_trl.py` 中）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_epochs` | 1 | 训练轮数 |
| `per_device_train_batch_size` | 1 | 批次大小 |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `learning_rate` | 1e-6 | 学习率（GSPO推荐较小值） |
| `num_generations` | 4 | 每个prompt生成的样本数 |
| `temperature` | 0.7 | 生成温度 |
| `beta` | 0.04 | KL惩罚系数 |

### 奖励函数

项目使用两种奖励函数：

1. **规则奖励** - 检查结构、格式、长度等
2. **DeepSeek API 奖励** - 调用外部模型评估内容质量

可通过 `.env` 中的 `USE_DEEPSEEK_JUDGE=false` 禁用 API 奖励。

## 📊 输出格式

### outline.jsonl 格式

```json
{
  "id": 1,
  "topic": "企业数字化转型战略规划",
  "outline": "# 企业数字化转型战略规划\n\n## 第一章...",
  "timestamp": "2025-12-20 22:51:37"
}
```

### content.jsonl 格式

```json
{
  "id": 1,
  "topic": "企业数字化转型战略规划",
  "outline": "# 企业数字化转型...",
  "generated_content": "# 企业数字化转型战略规划\n\n## 第一章 概述\n\n详细内容...",
  "content_length": 3500,
  "timestamp": "2025-12-21 10:30:00"
}
```

## 🖥️ GPU 要求

- **最低配置**: 单卡 16GB 显存（如 RTX 4080、A100 16GB）
- **推荐配置**: 单卡 24GB+ 显存（如 RTX 4090、A100 40GB）

模型会自动使用 `device_map="auto"` 进行多卡分配。

## 📈 监控训练

训练日志通过 WandB 记录。配置好 `WANDB_*` 环境变量后，可在 WandB 面板查看：
- 训练损失曲线
- 奖励分数变化
- 生成样本示例

## 📄 License

MIT License
