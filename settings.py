
#------------------ 路径设置 ------------------#
# 数据集目录
import torch


DATA_DIR="D:/Python_learning/新新闻摘要/dataset/"
# 模型参数/预训练模型目录
PARAM_DIR="D:/Python_learning/新新闻摘要/params/"
# 词频表地址
VOCAB_PATH="D:/Python_learning/新新闻摘要/dataset/vocab_cnt.pkl"
# 单词->数字
WORD_IDX_PATH="D:/Python_learning/新新闻摘要/dataset/word2idx.pkl"
# 数字->单词
IDX_WORD_PATH="D:/Python_learning/新新闻摘要/dataset/idx2word.pkl"

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
# 词典大小(拉满就不会出现UNK),注意输入至网络时要加4（还有四个特殊字符）
VOCAB_SIZE=10000
# 最长原文序列长度
MAX_SOURCE_LEN=2193
# 最长摘要序列长度
MAX_SUMMARY_LEN=587

# 限定序列长度（长于此长度做切割，短于此长度做padding）
SOURCE_THRESHOLD=1024
SUMMARY_THRESHOLD=550
# 读取数据时的标志
TRAIN_FALG=0
VAL_FALG=1
TEST_FALG=2
# 数据清理规则
# 顺序莫变！
PATTERNS_ONCE=[
    "by .*? published :.*?\. \| \..*? [0-9]+ \. ",
    "by \. .*? \. ",
    "-lrb- cnn -rrb- -- ",
    "\t(.*?-lrb- .*? -rrb- -- )",
    ]
PATTERNS_ANY=[
    "``|''"
    ]

#------------------ 其他设置 ------------------#
# 主要是设置在CPU跑还是GPU跑
DEVICE=torch.device("cuda:0")
EPOCHS=10
BATCH_SZIE=28


#------------------ 预训练模型设置 ------------------#

# 搜索束个数,保留最有前景的生成序列
NUM_BEAMS=5
# 预测序列最大长度
MAX_LEN=630
# 预测序列最小长度
MIN_LEN=20
