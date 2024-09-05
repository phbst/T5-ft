#!/bin/bash

# 设置测试参数
url="http://127.0.0.1:8081/generate"
data='{"inputs":"connect(filter, ","parameters":{"max_new_tokens":20}}'
concurrent_requests=125 # 并发请求数
total_requests=3000  # 总请求数

# 初始化状态码计数器
declare -A status_codes

# 记录开始时间（以秒为单位，带小数部分）
start_time=$(date +%s.%N)

# 使用并发请求测试吞吐量
echo "开始测试吞吐量..."
for i in $(seq 1 $concurrent_requests); do
    (
        for j in $(seq 1 $(($total_requests / $concurrent_requests))); do
            status_code=$(curl -X POST -d "$data" -H 'Content-Type: application/json' -o /dev/null -s -w "%{http_code}" $url)
            # 使用锁机制确保状态码计数器更新是线程安全的
            lock_file="/tmp/status_codes.lock"
            (
                flock -x 200
                ((status_codes[$status_code]++))
            ) 200>$lock_file
        done
    ) &
done

# 等待所有请求完成
wait

# 记录结束时间（以秒为单位，带小数部分）
end_time=$(date +%s.%N)

# 计算总耗时（以秒为单位，带小数部分）
elapsed_time=$(echo "$end_time - $start_time" | bc -l)

# 计算吞吐量 (requests per second)
throughput=$(echo "$total_requests / $elapsed_time" | bc -l)

echo "测试完成。"
echo "总请求数: $total_requests"
echo "总耗时: $elapsed_time 秒"
echo "吞吐量: $throughput 请求/秒"

# 输出状态码统计结果
echo "状态码统计:"
for code in "${!status_codes[@]}"; do
    echo "  $code: ${status_codes[$code]}"
done