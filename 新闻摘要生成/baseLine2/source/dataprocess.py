import os
from tqdm.std import tqdm
from settings import *
import json
import pickle as pkl
import re

def dataprocess(train_path="./新闻摘要生成/baseLine2/dataset/train_dataset.csv",test_path= "./新闻摘要生成/baseLine2/dataset/test_dataset.csv"):
    # 数据清洗
    def _clean(data):
        print("数据清洗开始...")
        clean_data = []
        for i, token in tqdm(enumerate(data)):
            res = token
            # 根据数据清理规则来进行数据清洗
            for pat in PATTERNS_ONCE:
                if ("\t" in pat):
                    res = re.sub(pat, "\t", res, 1)
                else:
                    res = re.sub(pat, "", res, 1)
            for pat in PATTERNS_ANY:
                res = re.sub(pat, "", res)
            clean_data.append(res)
        print("数据清洗完毕")
        return clean_data

    # 将处理后的数据保存为json文件
    def _save_to_Json(data, mode):
        if mode == 2:#测试数据
            for i in range(len(data)):
                source = data[i].split('\t')[1].strip('\n')
                if source != '':
                    dict_data = {"text": source, "summary": 'no summary'}
                    with open(new_test_path + str(i) + '.json', 'w+', encoding='utf8') as f:
                        f.write(json.dumps(dict_data, ensure_ascii=False))
        else:#训练数据
            for i in range(len(data)):
                if len(data[i].split('\t')) == 3:
                    source_seg = data[i].split("\t")[1]
                    traget_seg = data[i].split("\t")[2].strip('\n')
                    if source_seg and traget_seg != '':
                        dict_data = {"text": source_seg, "summary": traget_seg}
                        path = new_train_path
                        if mode == 1:#验证数据
                            path = new_val_path
                        with open(path + str(i) + '.json', 'w+', encoding='utf8') as f:
                            f.write(json.dumps(dict_data, ensure_ascii=False))

    with open(train_path, 'r', encoding='utf8') as f:
        train_data_all = f.readlines()
    with open(test_path, 'r', encoding='utf8') as f:
        test_data = f.readlines()

    train_data_all = _clean(train_data_all)
    test_data = _clean(test_data)

    train_data = train_data_all[:8000]  # 把训练集重新划分为训练子集和验证子集，保证验证集上loss最小的模型，预测测试集
    val_data = train_data_all[8000:]

    new_train_path = os.path.join("./新闻摘要生成/baseLine2/dataset/new_train/")
    new_val_path = os.path.join("./新闻摘要生成/baseLine2/dataset/new_val/")
    new_test_path = os.path.join("./新闻摘要生成/baseLine2/dataset/new_test/")

    os.makedirs(new_train_path, exist_ok=True)
    os.makedirs(new_val_path, exist_ok=True)
    os.makedirs(new_test_path, exist_ok=True)

    # 将数据文件读取之后，通过json文件保存
    _save_to_Json(train_data, TRAIN_FALG)
    _save_to_Json(val_data, VAL_FALG)
    _save_to_Json(test_data, TEST_FALG)

def count_files(path):
    count = 0
    for filename in os.listdir(path):
        file_path = os.path.join(path,filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            count += 1
    return count

def creat_VocabCounter(data_dir="./新闻摘要生成/baseLine2/dataset/"): #统计词汇建词频表
    from collections import Counter
    def getwords(path):
        datanum = count_files(path)
        # 摘要集合
        summary_words = []
        # 原文章集合
        source_words = []
        for i in range(datanum):
            if os.path.exists(os.path.join(path, f"{i}.json")):
                js_data = json.load(open(os.path.join(path, f"{i}.json"), encoding="utf8"))
                summary = ''.join(js_data['summary']).strip()
                summary_words.extend(summary.strip().split(' '))
                source = ''.join(js_data['text']).strip()
                source_words.extend(source.strip().split(' '))
        return source_words + summary_words
    vocab_counter = Counter()
    vocab_counter.update(t for t in getwords(data_dir + "new_train") if t != "")
    vocab_counter.update(t for t in getwords(data_dir + "new_val") if t != "")
    vocab_counter.update(t for t in getwords(data_dir + "new_test") if t != "")

    with open(VOCAB_PATH, "wb") as f:
        pkl.dump(vocab_counter, f)


def make_Vocab(vocab_size=VOCAB_SIZE): #建立词典
    # 读取词频表
    with open(VOCAB_PATH, "rb") as f:
        wc = pkl.load(f)
    idx2word = {}
    word2idx = {  # 创建单词到索引的映射
        PAD_WORD: 0, # <pad>:用于填充句子以达到固定长度
        UNK_WORD: 1,# <unk>：表示未知单词
        BOS_WORD: 2, # <bos>：句子的开始标记
        EOS_WORD: 3,  # <eos>：句子的结束标记
    }
    # 从词频表中取出vocab_size-4个最常见的单词，并将它们添加到word2idx中，索引从4开始因为之前4个索引已经被特殊单词标记
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w
    with open(WORD_IDX_PATH, "wb") as f:
        pkl.dump(word2idx, f)
    with open(IDX_WORD_PATH, "wb") as f:
        pkl.dump(idx2word, f)


if __name__ == '__main__':
    dataprocess()

    creat_VocabCounter()

    make_Vocab(VOCAB_SIZE)
