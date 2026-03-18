#!/bin/bash
# Run all NFPE benchmarks (GPU recommended).
# Usage: bash run_benchmarks.sh

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

echo "========================================="
echo "GPU check"
echo "========================================="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "========================================="
echo "PFI Benchmark: OU (d=2,5,10)"
echo "========================================="
python experiments/pfi_benchmark.py \
    --system ou \
    --dims 2 5 10 \
    --epochs 3000 \
    --batch-size 2000 \
    --log-interval 500

echo ""
echo "========================================="
echo "PFI Benchmark: Bistable (d=2,5,10)"
echo "========================================="
python experiments/pfi_benchmark.py \
    --system bistable \
    --dims 2 5 10 \
    --epochs 3000 \
    --batch-size 2000 \
    --sigma 0.5 \
    --t-end 1.0 \
    --log-interval 500

echo ""
echo "========================================="
echo "Timing Comparison (d=5, 500 epochs)"
echo "========================================="
python experiments/timing_comparison.py \
    --dim 5 \
    --epochs 500 \
    --log-interval 100

echo ""
echo "========================================="
echo "Timing Comparison (d=10, 500 epochs)"
echo "========================================="
python experiments/timing_comparison.py \
    --dim 10 \
    --epochs 500 \
    --log-interval 100 \
    --output-dir results/timing_d10

echo ""
echo "========================================="
echo "All done! Results in results/"
echo "========================================="
ls -la results/pfi_benchmark_ou/
ls -la results/pfi_benchmark_bistable/
ls -la results/timing/
ls -la results/timing_d10/
