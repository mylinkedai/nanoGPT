# nanoGPT 使用指南（训练 / 使用 / 微调）

本文以当前仓库为准，提供最小可用的训练、推理（使用）和微调流程。命令均在仓库根目录执行。

如果你在 Mac 上，推荐直接使用脚本：

- `./train_on_mac.sh`：默认使用 BPE + GPT-2 预训练权重进行微调（需要联网下载权重），自动准备数据并训练，训练后会跑一次示例推理
- `./interfere_on_mac.sh`：进入对话式推理（聊天）

## 1. 训练（从零训练）

### 1.1 准备数据

仓库内置了一些数据集准备脚本，以 `data/` 目录为入口。以 Shakespeare 字符级数据集为例：

```bash
python3 data/shakespeare_char/prepare.py
```

执行后会在 `data/shakespeare_char/` 生成 `train.bin`、`val.bin` 等二进制文件。

### 1.2 启动训练

使用配置文件启动训练（以字符级 Shakespeare 为例）：

```bash
python3 train.py config/train_shakespeare_char.py
```

训练产物默认写入 `out-shakespeare-char/`（如 `out-shakespeare-char/ckpt.pt`，由 `config/train_shakespeare_char.py` 的 `out_dir` 决定）。如需修改超参，请编辑对应的 `config/*.py`。

如果你使用 BPE + GPT-2 微调配置（推荐），训练产物默认在 `out-shakespeare-bpe/`，配置文件是 `config/finetune_shakespeare_gpt2.py`。

## 2. 使用（推理 / 生成文本）

训练完成后使用 `sample.py` 进行推理：

```bash
python3 sample.py --out_dir=out-shakespeare-bpe --device=mps
```

常用参数示例：

- `--start="Your prompt"`：指定提示词
- `--num_samples=3`：一次生成多个样本
- `--max_new_tokens=200`：控制生成长度

示例：

```bash
python3 sample.py --out_dir=out-shakespeare-bpe --device=mps --start="To be" --max_new_tokens=200
```

说明：

- `sample.py` 不支持 `--config=...` 参数；如果需要加载配置文件，请作为位置参数传入，例如 `python3 sample.py config/train_shakespeare_char.py --out_dir=out-shakespeare-char`。
- 没有 MPS 时请改成 `--device=cpu`。

### 2.1 对话式推理（聊天）

使用对话脚本进入聊天模式：

```bash
./interfere_on_mac.sh out-shakespeare-bpe
```

可用环境变量：

- `DEVICE`：`mps` 或 `cpu`
- `MAX_NEW_TOKENS`：单次回答长度
- `TEMPERATURE` / `TOP_K`：采样参数
- `HISTORY_TURNS`：保留历史轮数（默认 6）
- `SYSTEM_PROMPT`：系统提示词

## 3. 微调（Finetune）

微调的核心是：准备自己的数据、用合适的配置启动训练、并从已有权重继续训练。

### 3.1 准备你的数据

将数据准备成和已有脚本一致的 `train.bin` / `val.bin`。推荐做法：

1. 复制一个相近的数据准备脚本，例如 `data/shakespeare_char/prepare.py`（字符级）或 `data/shakespeare/prepare.py`（BPE 级）。
2. 按你的语料修改脚本，输出 `train.bin` 和 `val.bin`。
3. 在新目录下运行脚本，例如：

```bash
python3 data/your_dataset/prepare.py
```

### 3.2 使用已有权重继续训练

如果你要在已有模型上继续训练（微调），需要指定已有 checkpoint：

```bash
python3 train.py config/finetune_shakespeare_gpt2.py --init_from=resume --out_dir=out-shakespeare-bpe
```

说明：

- `--init_from=resume` 表示从 `out-shakespeare-bpe/ckpt.pt` 继续训练
- 如果你要从其它目录继续训练，设置 `--out_dir` 指向对应目录

### 3.3 常用微调调整点

- 减小学习率（例如 `learning_rate`）
- 增大或减小 `max_iters`
- 若数据量较小，建议增大 `eval_interval`，减少过拟合监控开销

这些都在配置文件里修改，例如：`config/train_shakespeare_char.py`。

## 4. 常见问题

- 训练/推理默认使用 GPU（如可用）。如需强制 CPU，可设置 `device=cpu`。
- 若显存不足，可调小 `batch_size` / `block_size`。
- 想让 `iter` 只占一行并持续更新，可用 `--log_inplace=True`（或 `LOG_INPLACE=1 ./train_on_mac.sh`）。
- 若使用 `init_from=gpt2*`，需要安装 `transformers` 并联网下载权重，脚本会自动处理。
- 训练日志默认写入 `train.log`（带时间戳）；可用 `LOG_FILE=xxx.log` 或 `LOG_TS=0` 关闭时间戳。

## 5. Mac 一键脚本说明

- `./train_on_mac.sh` 默认会自动准备数据、训练，并运行一次示例推理。
- 如需跳过示例推理，使用：`RUN_SAMPLE=0 ./train_on_mac.sh`
- 如需使用字符级配置，显式指定：`./train_on_mac.sh config/train_shakespeare_char.py`
- 如需 BPE 且不下载权重（从零训练），使用：`./train_on_mac.sh config/train_shakespeare_bpe_scratch.py`
- 训练时长预设（粗略估计）：`./train_on_mac.sh 10m`、`./train_on_mac.sh 2h`、`./train_on_mac.sh 1d`，或在配置后加时长：`./train_on_mac.sh config/finetune_shakespeare_gpt2.py 2h`。

---

如果你需要我根据你已有的数据目录或目标任务给出更精确的微调配置，告诉我你的数据规模、格式和目标模型大小。
