import time
import os
from torch import nn
from torch import optim
from torch.nn.modules.module import Module
from tqdm.std import tqdm
from settings import *
import json
import pickle as pkl
import re
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from rouge import Rouge
import models

# 数据处理一直到词典生成，都只需要生成一次即可
def Preprocess(train_path=DATA_DIR+"train_dataset.csv",test_path=DATA_DIR+"test_dataset.csv"):
    '''
    清理数据、划分验证集后重新保存至新文件
    '''
    
    # 数据清洗
    def _cleanData(data):
        print("数据清洗开始=========================================")
        
        clean_data=[]
        for i,d in tqdm(enumerate(data)):
            res=d
            for pat in PATTERNS_ONCE:
                if("\t" in pat):
                    res=re.sub(pat,"\t",res,1)
                else:
                    res=re.sub(pat,"",res,1)
            for pat in PATTERNS_ANY:
                res=re.sub(pat,"",res)
            
            clean_data.append(res)

        print("数据清洗完毕=========================================")
        return clean_data
    
    # 将处理后的数据保存为json文件
    def _save2Json(data,mode):
        
        
        if mode==2:
            
            for i in range(len(data)): 
                source=data[i].split('\t')[1].strip('\n')
                if source!='': 
                    dict_data={"text":source,"summary":'no summary'}#测试集没有参考摘要
                    
                    with open(new_test_path+str(i)+'.json','w+',encoding='utf-8') as f:
                        f.write(json.dumps(dict_data,ensure_ascii=False))
                    
        
        else:
            
            for i in range(len(data)):
                
                if len(data[i].split('\t'))==3:
                    source_seg=data[i].split("\t")[1]
                    traget_seg=data[i].split("\t")[2].strip('\n')
                    
                    
                    if source_seg and traget_seg !='':
                        dict_data={"text":source_seg,"summary":traget_seg}
                        path=new_train_path
                        if mode==1:
                            path= new_val_path  
                        with open(path+str(i)+'.json','w+',encoding='utf-8') as f:
                            f.write(json.dumps(dict_data,ensure_ascii=False)) 
                        

    
    with open(train_path,'r',encoding='utf-8') as f:
        train_data_all=f.readlines()

    with open(test_path,'r',encoding='utf-8') as f:
        test_data=f.readlines()
    
    # 数据清洗
    train_data_all=_cleanData(train_data_all)
    test_data=_cleanData(test_data)
    
    # 设置新文件路径
    new_train_path=os.path.join(DATA_DIR,"new_train/")
    new_val_path=os.path.join(DATA_DIR,"new_val/")
    new_test_path=os.path.join(DATA_DIR,"new_test/")

    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)

    if not os.path.exists(new_val_path):
        os.makedirs(new_val_path)

    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)

    train_data=train_data_all[:8000] #把训练集重新划分为训练子集和验证子集，保证验证集上loss最小的模型，预测测试集
    val_data=train_data_all[8000:]

    _save2Json(train_data,TRAIN_FALG)
    _save2Json(val_data,VAL_FALG)
    _save2Json(test_data,TEST_FALG)
    


def CountFiles(path):
    '''
    计算目标文件夹json文件数目
    '''
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data

def BuildVocabCounter(data_dir=DATA_DIR):
    '''
    统计所有词汇，建立词频表
    '''
    from collections import Counter
    
    def GetTokens(path):
        n_data=CountFiles(path)
        summary_words=[]
        source_words=[]
        for i in range(n_data):
            js_data=json.load(open(os.path.join(path,f"{i}.json"),encoding="utf-8"))
            # 遍历每一个json文件，提取出text字段和summary字段的内容，划分为单词
            summary=''.join(js_data['summary']).strip()
            summary_words.extend(summary.strip().split(' '))
            
            source=''.join(js_data['text']).strip()
            source_words.extend(source.strip().split(' '))

        return source_words+summary_words


    vocab_counter=Counter()
    vocab_counter.update(t for t in GetTokens(data_dir+"new_train") if t !="")
    vocab_counter.update(t for t in GetTokens(data_dir+"new_val") if t !="")
    vocab_counter.update(t for t in GetTokens(data_dir+"new_test") if t !="")

    with open(VOCAB_PATH,"wb") as f:
        pkl.dump(vocab_counter,f)

# 根据预先构建的词频表，生成词汇表
def MakeVocab(vocab_size=VOCAB_SIZE):
    '''
    建立词典，通过vocab_size设置字典大小，将常用词设置到字典即可，其他生僻词汇用'<unk>'表示
    '''
    with open(VOCAB_PATH,"rb") as f:
        wc=pkl.load(f)
    word2idx, idx2word = {}, {}
    word2idx[PAD_WORD] = 0
    word2idx[UNK_WORD] = 1
    word2idx[BOS_WORD] = 2
    word2idx[EOS_WORD] = 3
    # 得到词频最高的vocab_size个单词
    for i, (w, _) in enumerate(wc.most_common(vocab_size), 4):
        word2idx[w] = i
    for w, i in word2idx.items():
        idx2word[i] = w

    # 词到数字
    with open(WORD_IDX_PATH,"wb") as f:
        pkl.dump(word2idx,f)
    # 数字到词
    with open(IDX_WORD_PATH,"wb") as f:
        pkl.dump(idx2word,f)


# 找到训练集，验证集，测试集中最长序列的长度，用于后续的将所有序列填充到相同长度
def GetNumOfLongestSeq():
    '''
    找到最长的seq长度，用于padding
    '''
    
    def _findInFolders(path,length):
        max_len=0
        for i in range(length):
            js_data=json.load(open(os.path.join(path,f"{i}.json"),encoding="utf-8"))
            l_data=js_data["summary"].split(" ")
            l=len(l_data)
            if(max_len<len(l_data)):
                max_len=l
        return max_len
    
    train_path=os.path.join(DATA_DIR,"new_train/")
    val_path=os.path.join(DATA_DIR,"new_val/")
    test_path=os.path.join(DATA_DIR,"new_test/")

    train_length=CountFiles(train_path)
    val_length=CountFiles(val_path)
    test_length=CountFiles(test_path)
    
    return max(
        _findInFolders(train_path,train_length),
        _findInFolders(val_path,val_length),
        _findInFolders(test_path,test_length))

#  用于生成 PyTorch TensorDataset 的数据集类
class TextDataset(Dataset):
    '''生成TensorDataset'''
    def __init__(self,flag,word2id:dict):
        self.word2id=word2id
        self.path=DATA_DIR
        self.flag=flag
        if(flag==TRAIN_FALG):
            self.path+="new_train"
        elif(flag==VAL_FALG):
            self.path+="new_val"
        elif(flag==TEST_FALG):
            self.path+="new_test"
        else:
            raise Exception(f"No this flag:{flag}")
    
    def __len__(self):
        return CountFiles(self.path)

    def __getitem__(self, index):
        # 文本信息，注意，这里返回的都是一个列表，元素为被划分的一个个单词
        source=ReadJson2List(self.path,index)
        # 摘要信息
        summary=ReadJson2List(self.path,index,True)
        # 去掉summary中的空格和空字符串
        summary=[i for i in summary if (i!='' and i!=' ')]
        # 将source根据word2id字典转化为数字序列
        enc_x=[self.word2id[word] if word in self.word2id.keys() else UNK_NUM for word in source]
        # 对生成的数字序列进行填充或者截断，从而便于机器处理
        enc_x,enc_x_l=PaddingSeq(enc_x,SOURCE_THRESHOLD) 
        
        if(self.flag!=TEST_FALG):
            # 将summary根据word2id字典转化为数字序列
            dec_x=[self.word2id[word] if word in self.word2id.keys() else UNK_NUM for word in summary]
            # 为什么要加上EOS标签的原因是，对于训练序列到序列的模型，解码器需要学习如何生成一个完整的序列
            # decoder输入前面加上BOS、decoder的label最后加上EOS
            y=list(dec_x)
            y.append(EOS_NUM)
            # 进行填充规范化处理
            y,y_l=PaddingSeq(y,SUMMARY_THRESHOLD)

            # 对开始的序列也做同意处理
            dec_x.insert(0,BOS_NUM)
            dec_x,dec_x_l=PaddingSeq(dec_x,SUMMARY_THRESHOLD)
            # 返回值依次为：编码器输入，编码器输入有效长度，解码器输入，解码器输入有效长度，标签，标签有效长度
            return (torch.LongTensor(enc_x), enc_x_l), (torch.LongTensor(dec_x), dec_x_l), (torch.LongTensor(y), y_l)
        #  因为测试数集没有summary
        if(self.flag==TEST_FALG):
            return (torch.LongTensor(enc_x),enc_x_l)



# 填充文本序列，直接填充转换完的index列表
def PaddingSeq(line,threshold):
    p_len=len(line)
    # 大于规定长度就进行截断操作，小于规定长度就进行填充
    if(p_len>threshold):
        # 进行结束标记的更新，即使截断，最后也得是结束标记
        if(EOS_NUM in line):
            line[threshold-1]=EOS_NUM
        return line[:threshold],threshold
    # 若小于，则用PAD_NUM来进行填充
    return line + [PAD_NUM] * (threshold - len(line)),p_len

def ReadJson2List(dir,i,label=False):
    '''读取单个json文件（一个样本），并按空格分割转换成列表'''
    
    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))
    if label:
        return js_data["summary"].split(" ")
    return js_data["text"].split(" ")


# 评测最长公共子序列长度
def GetRouge(pred,label):
    '''获取ROUGR-L值'''
    rouge=Rouge()
    rouge_score = rouge.get_scores(pred, label)
    rouge_L_f1 = 0
    rouge_L_p = 0
    rouge_L_r = 0
    for d in rouge_score:
        # F1值、精确率和召回率
        rouge_L_f1 += d["rouge-l"]["f"]
        rouge_L_p += d["rouge-l"]["p"]
        rouge_L_r += d["rouge-l"]["r"]

    print("rouge_f1:%.2f" % (rouge_L_f1 / len(rouge_score)))
    print("rouge_p:%.2f" % (rouge_L_p / len(rouge_score)))
    print("rouge_r:%.2f" % (rouge_L_r / len(rouge_score)))
    return (rouge_L_f1 / len(rouge_score))
    


# 将数据样本转化为适合模型输入的张量格式
with open(WORD_IDX_PATH,"rb") as f:
        w2i=pkl.load(f)
#  shuffle为True表示在每一个epoch中，数据都会被随机打乱，以增加模型的泛化能力
# BATCH_SZIE是指每个batch包含的样本数量
train_iter=DataLoader(TextDataset(TRAIN_FALG,w2i),shuffle=True,batch_size=BATCH_SZIE,num_workers=8)
val_iter=DataLoader(TextDataset(VAL_FALG,w2i),shuffle=False,batch_size=BATCH_SZIE,num_workers=4)
test_iter=DataLoader(TextDataset(TEST_FALG,w2i),shuffle=False,batch_size=1)



# 模型训练：序列到序列的模型
def Train(net:Module,lr=0.01):
    from tqdm import tqdm

    # 初始化神经网络模型中的权重参数
    # 这种初始化方法有助于提高模型的训练收敛速度和性能
    def xavier_init_weights(m):
        # 判断当前的模型层是否是PyTorch中的nn.Linear层（全连接层）
        if type(m) == nn.Linear:
            # 使用 Xavier 均匀初始化方法初始化该层的权重参数 m.weight。
            # Xavier 均匀初始化方法能够帮助避免梯度消失或爆炸的问题
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # 准备训练环境
    # 神经网络模型进行了一系列的初始化和设置操做
    net.apply(xavier_init_weights)
    net.to(DEVICE)
    # 使用 Adam 优化算法创建了一个优化器 optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss = models.MaskedSoftmaxCELoss()
    
    # 验证集loss降到10000以下时开始保存每轮更低的参数
    # 损失反映了预测结果和真实结果的差距
    min_loss=10000
    for epoch in range(EPOCHS):
        train_loss=[]
        val_loss=[]
        # 设置训练模式
        net.train()
        # 调用进度条
        for batch in tqdm(train_iter):
            # 从一个 batch 的数据中,分别获取编码器的输入序列(enc_X)、编码器输入序列的有效长度(enc_x_l)、
            # 解码器的输入序列(dec_x)、解码器输入序列的有效长度(dec_x_l)以及标签序列(y)及其有效长度(y_l)
            (enc_X, enc_x_l), (dec_x, dec_x_l), (y,y_l) = [(x[0].to(DEVICE),x[1].to(DEVICE)) for x in batch]
            # 将编码器序列和解码器序列传递给神经网络进行向前推理，其会返回预测结果
            pred, _ = net(enc_X, dec_x, enc_x_l)
            # 计算预测结果pred与实际标签y之间的损失值，用交叉熵损失的方法
            l = loss(pred, y, y_l).sum()
            # 计算梯度
            l.backward()
            # 更新模型参数
            optimizer.step()
            # 清空上一个模型参数
            optimizer.zero_grad()

            # 得到训练集的损失值，停止梯度计算
            with torch.no_grad():
                train_loss.append(l.item())
            
        # 释放显存
        torch.cuda.empty_cache()

        # 将模式切换到评估模式
        net.eval()

        with torch.no_grad():
            for batch in tqdm(val_iter):
                (enc_X, enc_x_l), (dec_x, dec_x_l), (y,y_l) = [(x[0].to(DEVICE),x[1].to(DEVICE)) for x in batch]
                pred, _ = net(enc_X, dec_x, enc_x_l)
                l = loss(pred, y, y_l).sum()
                # 得到验证集的损失值
                val_loss.append(l.item())

        # 保存模型参数，秒级时间戳保证唯一性
        if(sum(val_loss)<min_loss):
            min_loss=sum(val_loss)
            torch.save(net.state_dict(),PARAM_DIR+str(int(time.time()))+"_GRU.param")
            print(f"saved net with val_loss:{min_loss}")
        # 打印每一轮训练，训练损失值和验证损失值的不同，观察是否过拟合或者欠拟合
        print(f"{epoch+1}: train_loss:{sum(train_loss)};val_loss:{sum(val_loss)}")
    

# 根据训练好的模型来生成结果
def GenSubmisson(net,param_path,max_steps=100):
    '''依据测试集，生成submission文件'''
    import csv
    with open(IDX_WORD_PATH,"rb") as f:
        i2w=pkl.load(f)
    
    net.load_state_dict(torch.load(param_path))
    net.eval()
    res=[]
    count=0
    for enc_X,enc_X_l in tqdm(test_iter):
        enc_X,enc_X_l=enc_X.to(DEVICE),enc_X_l.to(DEVICE)

        # 解码器的初始输入值张量
        dec_X = torch.unsqueeze(
            torch.tensor([BOS_NUM], dtype=torch.long, device=DEVICE),dim=0
            )
        # 使用预训练的net将输入序列进行编码
        enc_outputs = net.encoder(enc_X, enc_X_l)
        # 用预训练的net的解码器，根据编码输出来初始化解码器
        dec_state = net.decoder.init_state(enc_outputs, enc_X_l)
        
        output_seq, attention_weight_seq = [], []
        for _ in range(max_steps):
            # 使用解码网络,获得当前时间步的输出Y和更新后的解码状态
            Y, dec_state = net.decoder(dec_X, dec_state)
            # 选择概率最大的索引
            dec_X = Y.argmax(dim=2)
            # 获取预测的token索引pred,将其转化为标量整型
            pred = dec_X.squeeze(dim=0).type(torch.int32).item()
            if pred == EOS_NUM:
                break
            # 加入到输出序列
            output_seq.append(pred)
        # 根据i2w字典,进行数字序列到文本的转换
        pred_seq=' '.join([i2w[i] for i in output_seq])
        res.append([str(count),pred_seq])
        count+=1
    
    with open(os.path.join(DATA_DIR, 'submission.csv'),'w+',newline="",encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile,delimiter="\t")   
        writer.writerows(res)


if __name__=='__main__':
    # 下面所注释的代码，只需要执行一次得到结果即可
    # 数据预处理
    # Preprocess()
    # 建立词频表
    # BuildVocabCounter()
    # 建立词汇表
    # MakeVocab(VOCAB_SIZE)
    with open(WORD_IDX_PATH,"rb") as f:
        w2i=pkl.load(f)

    # 测试一个看看是否正常
    # a=TextDataset(VAL_FALG,w2i)
    # x=a.__getitem__(1)
    # 这是张量信息
    # print(x)

    # batch_size需要调整，不然gpu内存不够
    # train_iter=DataLoader(TextDataset(VAL_FALG,w2i),shuffle=True,batch_size=BATCH_SZIE,num_workers=4)
    # Train(models.GetTextSum_GRU(),0.01)
    #
    GenSubmisson(
        models.GetTextSum_GRU().to(DEVICE),
        os.path.join(PARAM_DIR,"1720090505_GRU.param")
        )
