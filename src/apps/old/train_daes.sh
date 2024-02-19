#!/bin/sh
# Usage: bash train_unets.sh --device CUDA_VISIBLE_DEVICE --iterations val1 val2 ...

CUDA_VISIBLE_DEVICES=""
iterations=()

while [ "$1" != "" ]; do
    case $1 in
        --device)
            CUDA_VISIBLE_DEVICES=$2
            shift # Remove --device
            shift # Remove the device value
            ;;
        --iterations)
            shift # Remove --iterations
            while [ "$1" != "" ]; do
                iterations+=($1)
                shift # Remove the iteration value
            done
            ;;
        *)
            echo "Error: Unknown argument $1"
            exit 1
            ;;
    esac
done

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES not set. Use --device flag."
    exit 1
fi

if [ ${#iterations[@]} -eq 0 ]; then
    echo "Error: No iterations specified. Use --iterations flag."
    exit 1
fi

for i in "${iterations[@]}"; do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_dae_calgary.py --iteration $i --residual 1
done