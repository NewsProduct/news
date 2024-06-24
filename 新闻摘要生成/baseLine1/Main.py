import csv
import re
import torch
import random
import numpy as np
import pandas as pd
from rouge import Rouge
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 复现种子设置函数，从而让实验结果可以复现
def set_seed(seed):
    # 为CPU设置种子用于生成随机数序列，使其每次程序运行都是一样的
    torch.manual_seed(seed)
    # 为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# 数据读取
def getInfo():
    # 测试集路径
    test_path = './test_dataset.csv'
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data  = f.readlines()
    # 创建一个DataFrame，增加Index和Text两列
    test = pd.DataFrame([], columns=["Index", "Text"])
    # 对text进行赋值
    for idx, rows in enumerate(test_data):
        test.loc[idx] = rows.split("\t")
    return test

# 处理文本摘要函数
def generateText(test,tokenizer,model):
    # 处理文本的空格字符和无用字符
    # 将连续空格替换为单个空格,然后将连续换行符替换为单个空格
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    # 存储所得的摘要集合
    info = []
    for idx, article_text in tqdm(enumerate(test["Text"][:multi_sample]), total=multi_sample):
        # 使用分词器将文本转化为模型可以理解的输入ID序列
        input_ids = tokenizer(

            [WHITESPACE_HANDLER(article_text)],

            return_tensors="pt",  # 转化为PyTorch张量模式

            padding="max_length",  # 确保输入长度不超过512

            truncation=True,

            max_length=512

        )["input_ids"].to(device)

        # 使用T5模型生成文本摘要
        output_ids = model.generate(

            input_ids=input_ids,

            max_length=512,  # 最大输出长度为512个token

            min_length=int(len(article_text) / 32),  # 设置最小输出长度

            no_repeat_ngram_size=3,  # 不允许连续3个相同的n-gram出现

            num_beams=5  # 设置使用5个beam搜素

        )[0]

        # 使用分词器将生成的输出id转化为可读的文本摘要
        summary = tokenizer.decode(

            output_ids,

            skip_special_tokens=True,  # 忽略掉特殊的token

            clean_up_tokenization_spaces=False

        )

        info.append(summary)
    return info

# 将生成的摘要结果写入文件
def writeResult(info):
    # 此次处理摘要的起始下标
    count = 750
    all_sample_test = []
    for i in info:
        all_sample_test.append([int(count), i])
        count += 1

    with open('submission1.csv', 'w+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerows(all_sample_test)


if __name__ == '__main__':
    # 设置种子
    set_seed(0)
    # 得到测试集数据
    test = getInfo()
    # 加载多语言T5模型
    model_name = "./PretrainModel/mT5_multilingual_XLSum"
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 将模型放到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    multi_sample = 250
    # 调用摘要生成函数
    info = generateText(test,tokenizer,model)
    # 将处理结果写入文件
    writeResult(info)