#!/bin/bash
set -e
echo "=== Building ==="
make clean && make

echo ""
echo "=== Testing CPU mode ==="
./leiden graph.txt cpu 1.0 > tests/current_cpu.txt 2>&1
grep -E "^(old |new |Leiden )" tests/baseline_cpu.txt > tests/expected_cpu_quality.txt
grep -E "^(old |new |Leiden )" tests/current_cpu.txt > tests/actual_cpu_quality.txt
if diff tests/expected_cpu_quality.txt tests/actual_cpu_quality.txt > /dev/null; then
    echo "CPU: PASS"
else
    echo "CPU: FAIL"
    diff tests/expected_cpu_quality.txt tests/actual_cpu_quality.txt
    exit 1
fi

echo ""
echo "=== Testing GPU mode ==="
./leiden graph.txt gpu 1.0 > tests/current_gpu.txt 2>&1
grep -E "^(previous |new |Leiden)" tests/baseline_gpu.txt > tests/expected_gpu_quality.txt
grep -E "^(previous |new |Leiden)" tests/current_gpu.txt > tests/actual_gpu_quality.txt
if diff tests/expected_gpu_quality.txt tests/actual_gpu_quality.txt > /dev/null; then
    echo "GPU: PASS"
else
    echo "GPU: FAIL"
    diff tests/expected_gpu_quality.txt tests/actual_gpu_quality.txt
    exit 1
fi

echo ""
echo "=== All tests passed ==="
