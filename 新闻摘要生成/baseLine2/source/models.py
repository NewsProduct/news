from torch import nn
import torch
from torch import Tensor
from settings import *
import utils

# 是nn.CrossEntropyLoss的子类，实现了带遮蔽的softmax交叉熵损失函数"
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    def _sequence_mask(self, X, valid_len, value=0):
        """ 在序列中屏蔽不相关的项。
            接收valid_len是多个有效长度组成的一维tensor，如[1，2]代表第一个序列有效长度为1，第二个序列有效长度为2
        """
        # 获取张量X的第二维大小，即每个样本的最大长度
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        # 将X中对应mask为False的元素全部赋值为value
        X[~mask] = value
        # 有效长度以外的元素都被置零，不改变原始shape
        return X

    # 实现向前传播
    def forward(self, pred, label, valid_len):
        # 不用看标签中的padding的损失
        weights = torch.ones_like(label)
        weights = self._sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)

        # 把整个序列的loss取平均，最后输出的shape是(batch_size)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


class Encoder(nn.Module):
    '''编码器接口'''
    def  __init__(self, **kwargs):
        super(Encoder,self).__init__(**kwargs)
    
    def forward(self,X,*args):
        raise NotImplementedError

class Decoder(nn.Module):
    '''编码器接口'''
    def  __init__(self, **kwargs):
        super(Decoder,self).__init__(**kwargs)
    
    # 接收编码器的输出，作为当前步的先验状态
    def init_state(self,enc_outputs,*args):
        raise NotImplementedError
    # state和解码器输入共同作为输入
    # 在一次序列训练中，初始state为编码器输入，之后会不断自我更新
    def forward(self,X,state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    '''编码器解码器架构基类'''
    def  __init__(self, encoder:Encoder,decoder:Decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,enc_X,dec_X,*args):
        enc_outputs=self.encoder(enc_X,*args)
        dec_state=self.decoder.init_state(enc_outputs)

        return self.decoder(dec_X,dec_state)


################################## RNN(循环神经网络)（效果太差了）下面是它的编码器和解码器
# GRU是一种特殊的循环神经网络单元，RNN 编码器的实现,它使用了 GRU 作为基本单元
class GruEncoder(Encoder):
    # in_dimL:输入特征维度；emb_dim:词嵌入维度；hidden_size:隐藏状态大小
    def __init__(self,in_dim,emb_dim,hidden_size,num_layers,dropout=0,**kwargs):
        super(GruEncoder,self).__init__(**kwargs)
        # nn.Embedding层，将输入序列映射到词嵌入向量
        self.embdding=nn.Embedding(in_dim,emb_dim)
        # nn.GRU层为RNN编码器，输入为词嵌入向量，输出为最终的隐藏状态
        self.rnn=nn.GRU(emb_dim,hidden_size,num_layers,dropout=dropout)

    # 接受输入序列X作为参数
    def forward(self,X:Tensor,*args):
        # 由嵌入层转化为词嵌入向量
        X=self.embdding(X)
        # 更改数据维度为seq_len,batch_size,features
        X=X.permute(1,0,2)
        output,state=self.rnn(X)
        # shape分别为：
        # (seq_len,batch_size,hidden_size)
        # (num_layers,batch_size,hidden_size)
        return output,state

# RNN解码器
class GruDecoder(Decoder):
    def __init__(self,in_dim,emb_dim,hidden_size,num_layers,dropout=0,**kwargs):
        super(GruDecoder,self).__init__(**kwargs)
        self.embdding=nn.Embedding(in_dim,emb_dim)
        # nn.GRU作为解码器，输入为词嵌入向量和上一时刻的隐藏状态，输出为当前时刻的输出和隐藏状态
        self.rnn=nn.GRU(emb_dim+hidden_size,hidden_size,num_layers,dropout=dropout)
        # nn.Linear层，将GRU的输出映射到最终的逻辑概率分布
        self.dense=nn.Linear(hidden_size,VOCAB_SIZE+4)

    # 接受编码器的输出enc_outputs
    def init_state(self, enc_outputs, *args):
        # 取enc的state，即编码器输出的最终的隐藏状态
        return enc_outputs[1]

    # X：当前时刻的输入序列X和上一时刻的隐藏状态state作为参数
    def forward(self,X:Tensor,state:Tensor):
        X=self.embdding(X).permute(1,0,2)
        # 取最后时刻的最后一层
        context=state[-1].repeat(X.shape[0],1,1)
        
        # 虽然state在h0已经传过来了，但是还是把state拼一下,拼到了特征的维度，问题不大
        X_and_context=torch.cat((X,context),2)
        # 得到当前时刻的输出和下一时刻的隐藏状态state
        output,state=self.rnn(X_and_context,hx=state)
        # 将output通过最终的self.dense线性层，得到当前时刻的逻辑概率分布
        output=self.dense(output).permute(1,0,2)
        # shape分别为：
        # (batch_size,seq_len,hidden_size)
        # (num_layers,batch_size,hidden_size)
        return output,state

# 返回一个基于GRU的编码器-解码器的模型
# VOCAB_SIZE+4：输入特征的维度，包括了词汇表大小以及其它的特殊标记
# 512: 词嵌入的维度
# 256：隐藏状态大小
# 2：编码器的层数
def GetTextSum_GRU():
    return EncoderDecoder(
        GruEncoder(VOCAB_SIZE+4,512,256,2),
        GruDecoder(VOCAB_SIZE+4,512,256,2)
    )
##################################



def GetModel(name:str):
    name=name.lower()
    if(name=="gru"):
        return GetTextSum_GRU().to(DEVICE)
    
    else:
        raise Exception("该模型未实现！")

if __name__=='__main__':
    # encoder=GruEncoder(VOCAB_SIZE+4,512,256,2)
    # decoder=GruDecoder(VOCAB_SIZE+4,512,256,2)
    # for enc_X,dec_X,y in utils.train_iter:
    #     print(enc_X[0].shape)
    #     enc_out=encoder(enc_X[0])
        
    #     state=decoder.init_state(enc_out)
    #     output,state=decoder(dec_X[0],state)
    #     print(output.shape)
    #     loss_f=MaskedSoftmaxCELoss()
    #     l=loss_f(output,y[0],y[1])
    #     print(l)
        
    #     break
        
    net=GetTextSum_GRU()
    

    with open("1.txt","w+") as f:
        f.write(str(net))
