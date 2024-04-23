API_RATE=50
OUTPUT_DIR=/nfs-data/user30/Projects/00MY/00DeStein/generations/results/gpt2-large/test

export CUDA_VISIBLE_DEVICES="4" 
python -m run_toxicity_experiment \
    --dataset-file /nfs-data/user30/Projects/00MY/00DeStein/data/RealToxicityPrompts/100/test.jsonl \
    --model-type gpt2-act \
    --model "/nfs-data/user30/model/gpt2-large" \
    --tokenizer "/nfs-data/user30/model/gpt2-large" \
    --perspective-rate-limit $API_RATE \
    --p 0.9 \
    --count 20 \
    --alpha 0.45 \
    --batch-size 25 \
    --n 25\
    --seed 42 \
    $OUTPUT_DIR
