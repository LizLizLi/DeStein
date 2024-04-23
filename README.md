# DESTEIN: Navigating Detoxification of Language Models via Universal Steering Pairs and Head-wise Activation Fusion
This repository contains code for the paper "DESTEIN: Navigating Detoxification of Language Models via Universal Steering Pairs and Head-wise Activation Fusion".

# Setup

Please use the command below to setup the environment needed.
```
conda create -n destein python=3.10
conda activate destein
pip install -r requirements.txt
```
# Data

To maintain the timeliness of the results, we used the [Perspective API](https://github.com/conversationai/perspectiveapi/tree/master/1-get-started) to rescore the toxicity of [Realtoxicityprompts](https://allenai.org/data/real-toxicity-prompts). Simultaneously, we sampled toxic (>=0.5) and non-toxic (<0.5) data as the test set, which are located in the ```./data/RealToxicityPrompts``` folder. 

The folder ```./data/act``` contains pre-calculated steering vectors and probe detection results. If you want to replicate the results in the paper, you can apply them directly.

# Detoxification

To generate continuations with DESTEIN and score them for toxicity using the PerspectiveAPI toxicity scorer, run the following command.

```
API_RATE=50
OUTPUT_DIR=generations/results/gpt2-large/test

export CUDA_VISIBLE_DEVICES="4" 
python -m run_toxicity_experiment \
    --dataset-file data/RealToxicityPrompts/100/test.jsonl \
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
```

In general, model_type is one of the base models(gpt2, llama2) and our methods(gpt2-act, llama2-act, opt-act, mpt-act). Different methods have different additional parameters to specify. For details, please refer to our paper.

This script will create three files in OUTPUT_DIR: generations.jsonl with all of the generated continuations, perspective.jsonl with all the scores from Perspective API, and prompted_gens_[model_type].jsonl, which collates the previous two files.

# Evaluation

To evaluate generated output for fluency and diversity, run the following command. The "generations_file" should have the format prompted_gens_[model_type].jsonl.

```
python -m eval.evaluate_generations \
    --generations_file $1
```

# Citation

```
@misc{li2024destein,
      title={DESTEIN: Navigating Detoxification of Language Models via Universal Steering Pairs and Head-wise Activation Fusion}, 
      author={Yu Li and Zhihua Wei and Han Jiang and Chuanyang Gong},
      year={2024},
      eprint={2404.10464},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
