# 使用预训练模型
import re

from transformers import PegasusTokenizer,PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration,AdamW
from transformers import BartTokenizer,BartForConditionalGeneration
from settings import *
import os

current_model=""

'''
函数接受4个参数:
net: 用于生成摘要的预训练神经网络模型。
tokenizer: 用于编码输入文本和解码生成的摘要的分词器。（将文本和机器可读的数字序列(token ID)相互转换）
text: 需要生成摘要的输入文本。
target(可选): 参考摘要,如果提供的话,将用于计算ROUGE评分。
'''
# 使用预训练的神经网络模型和分词器生成单个样本的摘要
def TestOneSeq(net,tokenizer,text, target=None):
    # 清空GPU内存缓存，确保后续操作有足够的内存空间
    torch.cuda.empty_cache()
    # 将模型切换到评估模式，关闭训练层
    net.eval()

    #定义lambda函数 WHITESPACE_HANDLER 来清理输入文本中的多余空格和换行
    '''
    参数 k是需要处理的字符串。
    第一个re.sub() 函数将字符串中的所有多个空白字符(包括空格和换行符)替换为单个空格。
    第二个re.sub() 函数将字符串中的所有多个换行符替换为单个空格。
    strip() 方法去除字符串开头和结尾的空白字符。
    '''
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    text = WHITESPACE_HANDLER(str(text))

    # 使用分词器对文本进行编码（将文本转换为机器可读的数字序列(token ID)），转换为pytorch张量.移动到 DEVICE 变量指定的设备上
    input_tokenized = tokenizer.encode(
        text,
        truncation=True, #输入文本的长度超过了预定义的最大长度,就对其进行截断
        return_tensors="pt",#返回的结果应该是一个PyTorch张量(tensor)
        max_length=SOURCE_THRESHOLD#设置分词器的最大输入长度阈值
        ).to(DEVICE)

    #如果当前使用的模型为 "t5"，创建一个PyTorch张量summary_task和之前编码的 input_tokenized 张量沿着最后一个维度(dim=-1)连接起来,形成一个新的张量
    if(current_model=="t5"):
        summary_task = torch.tensor([[21603, 10]]).to(DEVICE)#两个值代表了一个特殊的任务标记,用于告诉 T5 模型进行文本摘要生成任务
        input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(DEVICE)#将任务标记信息添加到输入数据的开头,作为模型的额外输入，移动到 DEVICE 上

    # 使用预训练模型的net.generate()方法生成摘要
    summary_ids = net.generate(input_tokenized,
                                    num_beams=NUM_BEAMS,   #用beam search算法生成摘要，并设置beam的数量为NUM_BEAMS
                                    no_repeat_ngram_size=3,    #禁止重复n-gram
                                    min_length=MIN_LEN,     #设置摘要的最小和最大长度
                                    max_length=MAX_LEN,
                                    early_stopping=True)    #达到摘要预期长度后停止生成

    # 用分词器将机器可读的数字序列(token ID)转换为文本
    # 注意，不清理文本中的空格
    '''
     遍历summary_ids 列表中的每个元素 g,并对每个元素调用分词器的decode()方法将单个token ID转换回原始的文本形式
     skip_special_tokens：解码过程中跳过特殊tokens,如开始/结束标记、填充标记等，确保输出的文本更加可读,不包含这些特殊的标记符号
     clean_up_tokenization_spaces：不要清理解码后的文本中的多余空格，确保输出文本的格式与原始文本更加一致
    '''
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    score=-1

    # 如果提供了参考摘要，使用 GetRouge() 函数计算生成的摘要与参考摘要之间的ROUGE评分
    if(target!=None):
        score=GetRouge(output[0],target)
    return output[0],score

# 用于加载预训练的 T5 模型和分词器,并返回它们以供后续使用
'''
name:指定要加载的特定模型和分词器的名称
PARAM_DIR:存储预训练模型和分词器的目录路径
'''
def GetTextSum_T5(name):
    tokenizer=T5Tokenizer.from_pretrained(PARAM_DIR+name)#使用 T5Tokenizer 类从指定路径 PARAM_DIR + name 加载预训练的分词器
    net=T5ForConditionalGeneration.from_pretrained(PARAM_DIR+name)#使用 T5ForConditionalGeneration 类从指定路径 PARAM_DIR + name 加载预训练的 T5 模型
    print(f"{name} 加载完毕")
    return net.to(DEVICE),tokenizer#加载好的 T5 模型移动到之前指定的计算设备 DEVICE 上并与分词器一起返回

#用于加载预训练的 BART 模型和分词器并返回它们以供后续使用
# bart专门用于条件文本生成任务，如摘要、翻译等等
def GetTextSum_BART():
    # 加载分词器
    tokenizer=BartTokenizer.from_pretrained(PARAM_DIR+"bart", output_past=True)
    # 加载模型
    net=BartForConditionalGeneration.from_pretrained(PARAM_DIR+"bart", output_past=True)
    print("bart 加载完毕")
    return (net.to(DEVICE),tokenizer)

#用于加载预训练的 Pegasus 模型和分词器并返回它们以供后续使用
def GetTextSum_Pegasus():
    tokenizer=PegasusTokenizer.from_pretrained(PARAM_DIR+"pegasus")
    net=PegasusForConditionalGeneration.from_pretrained(PARAM_DIR+"pegasus")
    print("pegasus 加载完毕")
    return (net.to(DEVICE),tokenizer)

#用于加载不同文本摘要模型的通用函数
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


#读取一个 JSON 格式的文件,并根据是否是测试文件返回相应的数据
#JSON 文件都包含了文本内容和相应的摘要信息每个 JSON 文件都包含两个键值对:"text": 存储文本内容  "summary": 存储对应的摘要
def ReadJson(i,dir,test=False):
    '''读取单个json文件（一个样本）'''
    import json

    '''
    json.load() 函数读取 JSON 文件的内容,并将其存储在 js_data 变量中
    os.path.join() 函数将目录 dir 和文件名 f"{i}.json" 拼接成一个完整的文件路径
    '''
    js_data=json.load(open(os.path.join(dir,f"{i}.json"),encoding="utf-8"))

    # 用于判断是读取测试文件还是训练集中的文件
    if test:
        return js_data["text"]#是读取测试文件，返回文本内容
    return js_data["text"],js_data["summary"]#不是读取测试文件，返回文本内容和摘要

#生成 submission.csv 文件，param_path：指定模型参数的路径
def GenSub(net,tokenizer,param_path=None):
    import csv
    from tqdm import tqdm
    
    if(param_path!=None):
        net.load_state_dict(torch.load(param_path))#使用 torch.load() 从指定文件加载模型参数,并更新 net 模型

    # 结果摘要集合
    res=[]#存储生成的摘要
    for i in tqdm(range(10)):
        #调用 ReadJson() 函数来读取测试文件中的文本, True 表示这是一个测试文件
        text=ReadJson(i,DATA_DIR+"new_test",True)
        #调用 TestOneSeq() 函数来使用 net 模型和 tokenizer 生成文本的摘要。
        summary=TestOneSeq(net,tokenizer,text)[0]
        #生成的摘要与文件索引 i 一起添加到 res 列表中
        res.append([int(i),summary])

    # 结果写入
    with open(os.path.join(DATA_DIR, 'submission.csv'),'w+',newline="",encoding='utf-8') as csvfile:
        writer=csv.writer(csvfile,delimiter="\t")   #将 res 列表写入文件,每个元素用制表符(\t)分隔
        writer.writerows(res)


if __name__=='__main__':
    net,tokenizer=GetPModel("bart")
    # # 生成submmision的函数，主要是通过调用不同模型，测试不同的结果
    GenSub(net,tokenizer)
