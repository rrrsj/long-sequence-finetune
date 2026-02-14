# long-sequence-finetune
---

# Qwen3-4B 分布式训练框架 (FSDP + DDP + Sequence Parallel)

这是一个基于 **PyTorch Distributed** 实现的 Llama 类模型（以 Qwen3-4B 为例）的高性能分布式训练方案。该项目融合了多种并行策略，旨在处理超长序列训练并优化显存占用。

## 🚀 核心特性

* **混合并行策略**：结合了 **FSDP (Fully Sharded Data Parallel)** 和 **DDP (Distributed Data Parallel)**。
* **序列并行 (Sequence Parallel)**：通过自定义 `SPAttentionGather` 和 `SPAttentionReduce` 算子，实现跨节点的序列切分与聚合，支持超长文本。
* **Monkey Patch 注入**：无需修改 Transformers 库源码，通过动态补丁替换 `Qwen3Attention` 的 `forward` 过程。
* **灵活的设备映射**：支持自定义进程组划分（例如：）。

---

## 📂 项目结构

| 文件 | 描述 |
| --- | --- |
| `train.py` | 训练主入口，负责环境初始化、模型加载与循环。 |
| `config.yaml` | 核心配置文件，包含模型路径、训练超参及并行配置。 |
| `model/sp_qwen3.py` | 模型包装类，负责 Monkey Patch 注入及架构初始化。 |
| `layer/sp_layer.py` | **核心实现**：包含基于 `torch.autograd.Function` 的 `all_to_all` 通讯算子。 |
| `utils/get_group.py` | 负责分布式进程组（Process Group）的逻辑划分。 |
| `utils/model_distribution.py` | 负责将模型分层包装进 FSDP 和 DDP 容器。 |

---

## 🛠️ 环境要求

* Python 3.10+
* PyTorch 2.0+ (支持 `device_mesh` 与 `FSDP`)
* Transformers 库
* NVIDIA GPUs (支持 NCCL 后端)

---

## ⚙️ 配置说明 (`config.yaml`)

你可以通过修改 `config.yaml` 来调整并行行为：

```yaml
training:
  load_type: "full"      # 加载类型: full (全量) 或 lazy (元数据)
  device_map: [2, 4]     # [DDP组大小, FSDP/序列并行组大小]
  max_length: 4096       # 最大序列长度
  batch_size: 2
  seed: 2026

```

> **注意**：`device_map[0] * device_map[1]` 必须等于总 GPU 数量。

---

## 🏃 如何运行

项目使用 `torchrun` 进行启动。默认配置为单机 8 卡：

1. **准备数据**：确保 GSM8K 数据集已放置在 `data/` 目录下。
2. **启动训练**：
```bash
bash run.sh

```


或者直接使用命令：
```bash
torchrun --nproc_per_node=8 train.py

```



---

## 💡 技术要点：序列并行实现

本项目在 Attention 层前后插入了通讯原语：

1. **Gather (Forward)**：在计算 Attention 前，通过 `all_to_all` 将分布在不同 GPU 上的序列片段重新分发，确保计算 Attention 时每个 Head 拥有完整的序列。
2. **Reduce (Forward)**：计算完成后，再次通过 `all_to_all` 将数据还原回原始的序列切分状态，以便进行后续的 MLP 层计算。
3. **梯度回传**：在 `sp_layer.py` 中手工实现了 `backward` 逻辑，确保梯度的跨设备一致性。

---

## 📝 日志查看

训练日志会根据 `rank` 自动保存至 `logs/` 文件夹：

* `logs/rank_0.log`
* `logs/rank_1.log`
* ...

---
