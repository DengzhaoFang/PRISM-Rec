# Checkpoint保存策略

## 问题说明

之前的实现中，即使设置了`--save_every 100`，在训练初期仍然每个epoch都保存checkpoint。

**原因**：训练初期loss持续下降，每个epoch都是`is_best=True`，触发checkpoint保存。

## 新的保存策略

### 三种Checkpoint类型

| 类型 | 文件名 | 保存条件 | 用途 |
|------|--------|---------|------|
| **Regular Checkpoint** | `checkpoint_epoch{N}.pt` | 每N个epoch（如100） | 定期备份，用于恢复训练 |
| **Best Model** | `best_model.pt` | 每次loss改进时 | 保存最佳模型 |
| **Latest Checkpoint** | `latest_checkpoint.pt` | 每个epoch（覆盖） | 快速恢复最新状态 |

### 保存逻辑

```python
# 每个epoch都会：
- 更新 latest_checkpoint.pt（覆盖）

# 当loss改进时：
- 更新 best_model.pt

# 每100个epoch（save_every）：
- 保存 checkpoint_epoch100.pt
- 保存 checkpoint_epoch200.pt
- ...
```

## 实际效果

### Before（旧版本）
```
训练初期（loss持续下降）：
Epoch 1  → checkpoint_epoch1.pt + best_model.pt + latest_checkpoint.pt
Epoch 2  → checkpoint_epoch2.pt + best_model.pt + latest_checkpoint.pt
Epoch 3  → checkpoint_epoch3.pt + best_model.pt + latest_checkpoint.pt
...
结果：每个epoch都保存3个文件（磁盘占用大）
```

### After（新版本）
```
训练初期（loss持续下降）：
Epoch 1   → best_model.pt + latest_checkpoint.pt
Epoch 2   → best_model.pt + latest_checkpoint.pt (覆盖)
Epoch 3   → best_model.pt + latest_checkpoint.pt (覆盖)
...
Epoch 100 → checkpoint_epoch100.pt + best_model.pt + latest_checkpoint.pt
Epoch 101 → best_model.pt + latest_checkpoint.pt (覆盖)
...
Epoch 200 → checkpoint_epoch200.pt + best_model.pt + latest_checkpoint.pt

结果：减少99%的定期checkpoint，节省磁盘空间
```

## 磁盘占用估算

假设每个checkpoint ~10MB，训练500 epochs：

| 策略 | Regular Checkpoints | 磁盘占用 |
|------|---------------------|---------|
| **旧版（save_every=50）** | ~500个 | ~5GB |
| **新版（save_every=100）** | 5个 | ~50MB |
| **节省** | 99% | 99% |

## 使用场景

### 1. 训练中断恢复

**恢复最新状态：**
```bash
--resume path/to/latest_checkpoint.pt
```

**恢复最佳模型：**
```bash
--resume path/to/best_model.pt
```

**恢复特定epoch：**
```bash
--resume path/to/checkpoint_epoch100.pt
```

### 2. 分析训练过程

如果需要分析不同训练阶段的模型，可以：
- 降低`save_every`（如50）
- 或在训练后期手动保存关键epoch

### 3. 磁盘空间有限

使用更大的`save_every`：
```bash
--save_every 200  # 或更大
```

## 配置建议

### 小数据集（<10万items）
```bash
--save_every 50   # 更频繁的备份
--epochs 200
```

### 中等数据集（10-100万items）
```bash
--save_every 100  # 默认配置
--epochs 500
```

### 大数据集（>100万items）
```bash
--save_every 200  # 减少保存频率
--epochs 1000
```

## 文件管理建议

### 训练完成后

保留：
- ✅ `best_model.pt` - 用于inference
- ✅ `semantic_id_mappings.json` - 最终结果
- ✅ `semantic_id_analysis.json` - 分析报告
- ✅ `training.log` - 训练日志

可删除：
- 🗑️ `checkpoint_epoch*.pt` - 如果不需要恢复训练
- 🗑️ `latest_checkpoint.pt` - 训练结束后不需要

### 自动清理脚本

```bash
# 清理中间checkpoint，只保留best和final
cd output_dir
rm -f checkpoint_epoch[0-9]*.pt
rm -f latest_checkpoint.pt
# 保留 best_model.pt 和最后一个checkpoint
```

## FAQ

### Q: 为什么best_model.pt还是频繁更新？

A: 这是正常的！在训练初期，loss持续下降，best model需要实时更新。只有regular checkpoint（`checkpoint_epoch*.pt`）才会按照`save_every`控制。

### Q: 如何完全禁用regular checkpoint？

A: 设置一个很大的值：
```bash
--save_every 99999
```

### Q: latest_checkpoint.pt有什么用？

A: 它每个epoch都会覆盖更新，用于：
- 训练意外中断后快速恢复
- 不占用额外磁盘空间（只有1个文件）
- 总是包含最新的训练状态

### Q: 如果训练中途想保存当前状态怎么办？

A: 可以：
1. 等待下一个`save_every`的epoch
2. 或者发送SIGTERM信号让训练优雅退出（会保存final checkpoint）
3. 或者手动复制`latest_checkpoint.pt`

---

**更新时间**: 2025-11-11
**版本**: v2

