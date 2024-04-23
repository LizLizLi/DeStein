export CUDA_VISIBLE_DEVICES="1" 
python -m eval.evaluate_generations \
    --generations_file $1
