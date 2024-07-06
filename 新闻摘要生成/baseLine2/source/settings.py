
#------------------ 路径设置 ------------------#
# 数据集目录
import torch


DATA_DIR="D:/Python_learning/新闻摘要生成/baseLine2/dataset/"
# 模型参数/预训练模型目录
PARAM_DIR="D:/Python_learning/新闻摘要生成/baseLine2/params/"
# 词频表地址
VOCAB_PATH="D:/Python_learning/新闻摘要生成/baseLine2/dataset/vocab_cnt.pkl"
# 单词->数字
WORD_IDX_PATH="D:/Python_learning/新闻摘要生成/baseLine2/dataset/word2idx.pkl"
# 数字->单词
IDX_WORD_PATH="D:/Python_learning/新闻摘要生成/baseLine2/dataset/idx2word.pkl"

#------------------ 词典设置 ------------------#
# 特殊符号
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_NUM = 0
UNK_NUM = 1
BOS_NUM = 2
EOS_NUM = 3
# 限定序列长度（长于此长度做切割，短于此长度做padding）
SOURCE_THRESHOLD=1024
SUMMARY_THRESHOLD=550

# 词典大小(拉满就不会出现UNK),注意输入至网络时要加4（还有四个特殊字符）
VOCAB_SIZE=10000
# 最长原文序列长度
MAX_SOURCE_LEN=2193
# 最长摘要序列长度
MAX_SUMMARY_LEN=587
# 读取数据时的标志
TRAIN_FALG=0
VAL_FALG=1
TEST_FALG=2
# 数据清理规则
# 顺序莫变！
# 会匹配类似"by John Doe published: Some text. | .2023 ."这样的文本,并将其替换掉
# 这些模式通常用于清洗一些不必要的文本信息,如作者信息、发布日期等。
PATTERNS_ONCE=[
    "by .*? published :.*?\. \| \..*? [0-9]+ \. ",
    "by \. .*? \. ",
    "-lrb- cnn -rrb- -- ",
    "\t(.*?-lrb- .*? -rrb- -- )",
    ]
# 这模式通常用于清洗一些标点符号或者特殊字符等
PATTERNS_ANY=[
    "``|''"
    ]

#------------------ 其他设置 ------------------#
# 主要是设置在CPU跑还是GPU跑
DEVICE=torch.device("cuda:0")
EPOCHS=3
BATCH_SZIE=1


#------------------ 预训练模型设置 ------------------#

# 搜索束个数,保留最有前景的生成序列
NUM_BEAMS=4
# 预测序列最大长度
MAX_LEN=142
# 预测序列最小长度
MIN_LEN=56

# 在保持其它参数不变的情况下，加大搜索束个数，ROUGE_L的分数会更高