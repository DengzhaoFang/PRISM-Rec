#!/bin/bash

# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示开始信息
echo "倒计时开始，3小时后将执行 train_recommender.sh"
echo "开始时间: $(date)"
echo "预计执行时间: $(date -d "+3 hours" "+%Y-%m-%d %H:%M:%S")"
echo "脚本将在后台运行，你可以关闭终端或执行其他操作"
echo "如果要停止倒计时，可以使用: pkill -f \"$(basename "$0")\""

# 等待3小时（10800秒）
#sleep 10800
sleep 10800

# 执行训练脚本
echo "时间到！开始执行 train_recommender.sh..."
echo "执行时间: $(date)"
cd "$SCRIPT_DIR" && bash train_recommender.sh