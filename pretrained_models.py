# 使用预训练模型
import re

from transformers import PegasusTokenizer,PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
from transformers import BartTokenizer,BartForConditionalGeneration
from settings import *
from utils import GetRouge,CountFiles
import os
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.module import Module

current_model=""



def ToTensor(texts,summaries,tokenizer):
    task_prefix="summarize: "
    encoding = tokenizer([task_prefix + sequence for sequence in texts], 
                    padding='longest', 
                    max_length=SOURCE_THRESHOLD, 
                    truncation=True, 
                    return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    target_encoding = tokenizer(summaries, 
                        padding='longest', 
                        max_length=SUMMARY_THRESHOLD, 
                        truncation=True)
    labels = target_encoding.input_ids
    labels = [(i if i != tokenizer.pad_token_id else -100) for i in labels]
    labels = torch.tensor(labels)

    return TensorDataset(input_ids,attention_mask,labels)

# 用于微调给定的文本摘要模型
def FineTune(net:Module,tokenizer):
    '''微调'''
    
    tset_texts=[]
    tset_summaries=[]
    vset_texts=[]
    vset_summaries=[]
    tset_len=CountFiles(DATA_DIR+"new_train")
    vset_len=CountFiles(DATA_DIR+"new_val")
    for i in range(tset_len):
        text,summary=ReadJson(i,DATA_DIR+"new_train")
        tset_texts.append(text)
        tset_summaries.append(summary)
    for i in range(vset_len):
        text,summary=ReadJson(i,DATA_DIR+"new_val")
        vset_texts.append(text)
        vset_summaries.append(summary)
    print("训练数据已读入内存...")    

    train_iter=DataLoader(
        ToTensor(tset_texts,tset_summaries,tokenizer),
        batch_size=BATCH_SZIE,
        shuffle=True,
        num_workers=4
        )
    val_iter=DataLoader(
        ToTensor(vset_texts,vset_summaries,tokenizer),
        batch_size=BATCH_SZIE,
        shuffle=False,
        num_workers=4
        )

    print("minibatch已生成...") 
       
    print("开始训练模型...")    
    opt=AdamW(net.parameters())
    from tqdm import tqdm
    import time
    min_loss=10
    for epoch in range(EPOCHS):
        train_loss=[]
        val_loss=[]
        net.train()
        for batch in tqdm(train_iter):
            input_ids,attention_mask,labels=[x.to(DEVICE) for x in batch]
            l = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            l.backward()
            opt.step()       
            opt.zero_grad()
            with torch.no_grad():
                train_loss.append(l.item())
        
        torch.cuda.empty_cache()
        net.eval()
        with torch.no_grad():
            for batch in tqdm(val_iter):
                input_ids,attention_mask,labels=[x.to(DEVICE) for x in batch]
                l = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                val_loss.append(l.item())
        
        if(sum(val_loss)<min_loss):
            min_loss=sum(val_loss)
            torch.save(net.state_dict(),PARAM_DIR+str(int(time.time()))+"_GRU.param")
            print(f"saved net with val_loss:{min_loss}")    
        
        print(f"{epoch+1}: train_loss:{sum(train_loss)};val_loss:{sum(val_loss)}")

def TestOneSeq(net,tokenizer,text, target=None):
    '''生成单个样本的摘要'''
    # 清空GPU内存缓存，确保后续操作有足够的内存空间
    torch.cuda.empty_cache()
    # 将模型切换到评估模式，关闭训练层
    net.eval()
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    text = WHITESPACE_HANDLER(str(text))
    # 使用分词器对文本进行编码，转换为pytorch张量
    input_tokenized = tokenizer.encode(
        text,
        truncation=True, 
        return_tensors="pt",
        max_length=SOURCE_THRESHOLD
        ).to(DEVICE)
    
    if(current_model=="t5"):
        summary_task = torch.tensor([[21603, 10]]).to(DEVICE)
        input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(DEVICE)

    # 使用预训练模型生成摘要
    summary_ids = net.generate(input_tokenized,
                                    num_beams=NUM_BEAMS,   # 用beam search算法生成摘要，并设置beam的数量为NUM_BEAMS
                                    no_repeat_ngram_size=3,
                                    min_length=MIN_LEN,
                                    max_length=MAX_LEN,
                                    early_stopping=True)    # 达到摘要预期长度后停止生成
    # 用分词器将生成的摘要id解码为可读的文本格式
    # 注意，不清理文本中的空格
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    score=-1
    # 是否提供了参考摘要
    if(target!=None):
        score=GetRouge(output[0],target)
    return output[0],score

# t5
def GetTextSum_T5(name):
    tokenizer=T5Tokenizer.from_pretrained(PARAM_DIR+name)
    net=T5ForConditionalGeneration.from_pretrained(PARAM_DIR+name)
    print(f"{name} 加载完毕")
    return net.to(DEVICE),tokenizer

# bart专门用于条件文本生成任务，如摘要、翻译等等
def GetTextSum_BART():
    # 加载分词器
    tokenizer=BartTokenizer.from_pretrained(PARAM_DIR+"bart", output_past=True)
    # 加载模型
    net=BartForConditionalGeneration.from_pretrained(PARAM_DIR+"bart", output_past=True)
    print("bart 加载完毕")
    return (net.to(DEVICE),tokenizer)

def GetTextSum_Pegasus():
    tokenizer=PegasusTokenizer.from_pretrained(PARAM_DIR+"pegasus")
    net=PegasusForConditionalGeneration.from_pretrained(PARAM_DIR+"pegasus")
    print("pegasus 加载完毕")
    return (net.to(DEVICE),tokenizer)

def GetPModel(name:str):
    global current_model
    name=name.lower()
    print("正在加载模型")
    if("t5" in name):
        current_model="t5"
        return GetTextSum_T5(name)
    elif(name=="bart"):
        return GetTextSum_BART()
    elif(name=="pegasus"):
        current_model="pegasus"
        return GetTextSum_Pegasus()
    else:
        raise Exception("该模型未实现！")
    
def ReadJson(i,dir,test=False):
    '''读取单个json文件（一个样本）'''
    import json

    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))
    # 用于判断是读取测试文件还是训练集中的文件
    if test:
        return js_data["text"]
    return js_data["text"],js_data["summary"]

def GenSub(net,tokenizer,param_path=None):
    '''生成submission.csv'''
    import csv
    from tqdm import tqdm
    
    if(param_path!=None):
        net.load_state_dict(torch.load(param_path))
    # 结果摘要集合
    res=[]
    for i in tqdm(range(1000)):
        text=ReadJson(i,DATA_DIR+"new_test",True)
        summary=TestOneSeq(net,tokenizer,text)[0]
        res.append([int(i),summary])

    # 结果写入
    with open(os.path.join(DATA_DIR, 'submission.csv'),'w+',newline="",encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile,delimiter="\t")   
        writer.writerows(res)


if __name__=='__main__':
    net,tokenizer=GetPModel("bart")
    # 指定输出为pyTorch张量
    # padding：指定序列填充
    # res=tokenizer(
    #     ["hello world","hi"],
    #     return_tensors="pt",
    #     padding='longest',
    #     max_length=MAX_LEN,
    #     truncation=True,
    #     )
    # # 打印结果张量
    # print(res)

    # # 生成submmision的函数，主要是通过调用不同模型，测试不同的结果
    GenSub(net,tokenizer)

    # 优化器
    opt=AdamW(net.parameters())
    opt.step()

    FineTune(net,tokenizer)
    
    with open("1.txt","w+") as f:
        f.write(str(net))
