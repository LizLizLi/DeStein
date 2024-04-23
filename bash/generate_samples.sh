OUTPUT_DIR=/nfs-data/user30/Projects/00MY/00DeStein/generations/samples_conditional/opt
export CUDA_VISIBLE_DEVICES="4" 
python -m run_samples_experiment \
    --use-dataset \
    --dataset-file /nfs-data/user30/datasets/condition-paradetoxic/output.jsonl \
    --model-type opt \
    --model "/nfs-data/user30/model/opt-6.7b" \
    --p 0.9 \
    --batch-size 8 \
    --n 1 \
    --max-tokens 100  \
    --resume \
    --seed 43 \
    $OUTPUT_DIR