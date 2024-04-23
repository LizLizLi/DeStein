import csv
import random
import json


def extract_random_elements(input_list, count, seed):
    """
    A specified number of elements are randomly selected from the given list.
    """
    number_of_elements = min(len(input_list), count)
    random.seed(seed)
    selected_elements = random.sample(input_list, number_of_elements)
    return selected_elements


def extract_select_elements(count, seed):
    para_detox = []

    with open(
        "/nfs-data/user30/Projects/adapter-gpt2/data/paraDetox/paradetox_processing_select.jsonl",
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            data = json.loads(line)
            text_list = data.get("text", [])
            text_list[0] = (
                "The following text is abusive, harmful, negative, obscene, racist, rude and toxic: "
                + text_list[0]
            )
            text_list[1] = (
                "The following text is kind, polite, positive, respectful and supportive: "
                + text_list[1]
            )
            sentences_tuple = tuple(text_list)
            para_detox.append(sentences_tuple)

    selected_elements = extract_random_elements(para_detox, count, seed)
    return selected_elements


def get_data(count, seed):
    with open(
        "/nfs-data/user30/Projects/adapter-gpt2/data/paraDetox/nopara/gpt2-large/500/final_para.jsonl",
        "r",
        encoding="utf-8",
    ) as file:
        paraDetox = []
        for line in file:
            data = json.loads(line)
            text_list = []
            text_list.append(
                "The following text is abusive, harmful, negative, obscene, racist, rude and toxic: "
                + data[0]
            )
            text_list.append(
                "The following text is kind, polite, positive, respectful and supportive: "
                + data[1]
            )
            sentences_tuple = tuple(text_list)
            paraDetox.append(sentences_tuple)

        paraDetox = extract_random_elements(paraDetox, count=count, seed=seed)
    # # 打开一个jsonl文件用于写入
    # with open(
    #     "/nfs-data/user30/Projects/adapter-gpt2/data/paraDetox/nopara/llama2-7b/final.jsonl",
    #     "w",
    # ) as f:
    #     for para in paraDetox:
    #         # 将每个元组转换为一个列表,然后使用json.dumps()将列表转换为JSON字符串
    #         cleaned_para = [
    #             s.replace(
    #                 "The following text is abusive, harmful, negative, obscene, racist, rude and toxic: ",
    #                 "",
    #             )
    #             for s in list(para)
    #         ]
    #         cleaned_para = [
    #             s.replace(
    #                 "The following text is kind, polite, positive, respectful and supportive: ",
    #                 "",
    #             )
    #             for s in list(cleaned_para)
    #         ]
    #         json_string = json.dumps(cleaned_para)
    #         # 将JSON字符串写入文件,后面加上换行符
    #         f.write(json_string + "\n")
    return paraDetox
