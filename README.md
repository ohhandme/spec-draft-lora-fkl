# Draft内部LoRA：复用现有5000条数据

当前执行入口是 `ce_existing_data.py`。目标是比较冻结Draft加修复头，与在Draft的q_proj/v_proj内部学习LoRA增量、直接改变起草分布。训练只读已保存的目标token，不采集新的Target分布；历史FKL入口见 `README_fkl_legacy.md`，不在当前排程执行。

## 输入和训练预算

挂载原有模型清单和权重、RAW5000派生的训练索引：

```text
/mnt/spec-inputs/
  models.json
  weights/Qwen3-0.6B/
  weights/Qwen3-8B/
  train/index.jsonl
```

索引SHA256锁为 `6b8797ea5155e4fd9bd01474fd3c669e6902e2025b5723e03e2a5faf311c3ff6`。训练3986请求、156688个既有窗口；校准503请求、19736窗口。标签直接读窗口中原有的Target token，实际前驱是对应的已确认参考序列。训练需要重新计算Draft前向以传播到内部LoRA，不能把冻结Draft的旧隐藏量当作可微训练输入；这一步没有Target模型或新教师标签。

两臂分别用每位等权和8→1前位权。每窗口按其有效位置权重和归一，再对32窗口取平均。分组microbatch累积同一个32窗口均值，末批按实际窗口数归一。两臂seed42、q/v LoRA rank8 alpha16、AdamW lr1e-4/weight_decay0.01、5个epoch，每臂24485次优化更新，样本与标签暴露预算相同。梯度只检查有限性，不截断，与原first4头的规则相同。

该预算对齐已有first4 MLP/单层Transformer CE头的32窗、5epoch条件。它与后来batch1的783440更新矩阵不是同一个优化器预算。LoRA与修复头的反馈输入、参数量仍有差别，结果回答两套方案在这套训练条件下哪个更有效，不是单一容量变量的比较。

## 检查和执行

```bash
python -m pip install -r requirements.txt
python cpu_ce_smoke.py
for loss in equal front; do
  python ce_existing_data.py smoke --assets /mnt/spec-inputs --out /srv/spec-runs/lora-ce --loss "$loss"
  python ce_existing_data.py train --assets /mnt/spec-inputs --out /srv/spec-runs/lora-ce --loss "$loss"
  python ce_existing_data.py eval --assets /mnt/spec-inputs --out /srv/spec-runs/lora-ce --loss "$loss"
  python ce_existing_data.py compare --assets /mnt/spec-inputs --out /srv/spec-runs/lora-ce --loss "$loss"
done
```

同一机器的所有入口传同一个 `--gpu-lock`。原机用 `/data/a100-first-rejection-20261111-9b21f68f/a1_gpu.lock`。先停止并退役旧的教师采集监督器，保留已有分片，再登记本入口；不要同时跑旧collect队列。

`smoke`只加载0.6B Draft，用现有503题中的3条原始RAW轨迹核对native与零LoRA候选，并执行一次现成标签的梯度更新。`train`也只加载Draft；断点记录窗口位置、优化器、总监督token和训练历史，可从已保存更新恢复。等权和前位权共享同一初始化与epoch顺序。训练结束按各自窗口权重的503请求等权校准CE选一个checkpoint。

## 结果口径

`inputs/development503_existing.jsonl`由已封存的A100503原件派生，包含原有prompt、Target答案和RAW/first4头逐题计数；来源见 `inputs/existing503_source.json`。不重新生成答案。评估使用原生 `draft.next(prefix,None,position)`，Target仅执行正常的推理验证，oracle断言保留。`compare`从LoRA逐轮原件复算接受数/验证轮，和已有同题RAW及对应权重的CE头作题族配对bootstrap。

503题用于开发和校准，不能写成独立确认。历史RAW耗时也不能充当新的同步测速对照；当前报告包含LoRA实际耗时，不据此单独宣称端到端加速。
