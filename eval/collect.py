import json

# 为 gpt4 验证准备数据


def extract_data(input_file_path, output_file_path):
    # 打开原始文件和输出文件
    with open(input_file_path, "r", encoding="utf-8") as input_file, open(
        output_file_path, "w", encoding="utf-8"
    ) as output_file:
        # 遍历原始文件的每一行
        for i, line in enumerate(input_file, start=1):
            # 如果行号是1或25的倍数（即第1, 26, 51, ...行），则写入新文件
            if i % 25 == 1:
                output_file.write(line)


def merge(extracted_file_path, prompt_file_path, merged_file_path):
    with open(extracted_file_path, "r", encoding="utf-8") as extracted_file, open(
        prompt_file_path, "r", encoding="utf-8"
    ) as another_file, open(merged_file_path, "w", encoding="utf-8") as merged_file:
        # 同时遍历两个输入文件
        for extracted_line, another_line in zip(extracted_file, another_file):
            another_line = json.loads(another_line)
            # 合并两个文件的每一行
            merged_line = (
                '"'
                + another_line["prompt"]["text"].strip()
                + extracted_line.strip().strip('"')
                + '"'
                + "\n"
            )
            # 将合并后的行写入输出文件
            merged_file.write(merged_line)


if __name__ == "__main__":
    ori = "/nfs-data/user30/Projects/adapter-gpt2/generations/toxicity/new-prompts/test/llama2-7b-lma"
    input_file_path = ori + "/generations.jsonl"  # 原始文件路径
    output_file_path = ori + "/generations_gpt4_eva.jsonl"  # 输出文件路径
    prompt_file_path = "/nfs-data/user30/datasets/RealToxicityPrompts/data/perspective-2024-1-13/100/test.jsonl"
    merged_file_path = ori + "/generations_gpt4_merge.jsonl"
    # 抽取原文件数据 每 25 行抽一行
    # extract_data(input_file_path, output_file_path)
    merge(output_file_path, prompt_file_path, merged_file_path)
