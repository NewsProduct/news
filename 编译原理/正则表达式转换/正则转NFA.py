
# 符号表
from 编译原理.正则表达式转换.DFA最小化 import MMFA
from 编译原理.正则表达式转换.NFA转DFA import NFA_DFA
from 编译原理.正则表达式转换.中缀转后缀 import ZhongZuiToHouZui
from 编译原理.正则表达式转换.后缀式转NFA import NFA

ch = ['(', ')', '*', '|']

class zhengZetoNFA:

    erro = []
    def __init__(self,strs):
        self.strs = strs

    # 判断字符是否合法
    def checkCharacter(self,str):
        for i in str:
            if i.isalpha() or i in ch:
                continue
            else:
                return False
        return True

    # 判断括号是否匹配
    def checkKuohaoPipei(self,str):
        a = 0
        b = 0
        number1 = 0
        for i in range(0,len(str)):
            if str[i] == '(':
                a += 1
                number1 = i
            elif str[i] == ')':
                b+=1
                # 由于是右括号，因此右括号的位置肯定不能大于左括号
                if b>a:
                    return False
                if i == number1+1:
                    return False
        if a==b:
            return True
        else:
            return False

    # 先判断正规式是否合法
    def judgeIsLegal(self,str):
        if self.checkCharacter(str) and self.checkKuohaoPipei(str):
            return True
        else:
            return False

    # 在表达式中加·，便于后续变后缀表达式
    def add_point(self, str):
        str0 = ''
        for i in range(0,len(str)-1):
            first = str[i]
            second = str[i+1]
            str0 += first
            if first != '(':
                if first != '|':
                    if second.isalpha():
                        str0 += '·'
            if second == '(':
                if first != '|' :
                    if first != '(':
                        str0 += '·'
        # 因为最后可能会str0可能会多产生一个 .
        str0 += str[len(str)-1]
        return str0

    def generate(self,str):
        if self.judgeIsLegal(str):
            # 在表达式中加·，便于后续变后缀表达式
            str = self.add_point(str)
            # 转为后缀表达式
            tmp1 = ZhongZuiToHouZui(str)
            str1 = tmp1.getResult()
            # # 表达式转NFA
            mm = NFA(str1)
            NFA_result, start,end = mm.express_NFA()
            print('NFA:')
            print(NFA_result)
            print("起始节点：")
            print(start)
            print('终止节点:')
            print(end)
            return NFA_result, start, end
        else:
            self.erro.append("正则表达式有错误")
            return None,None,None

if __name__ == '__main__':
    mk = zhengZetoNFA("ab*|b")
    # 得到NFA
    NFA_result, start, end = mk.generate("ab*|b")
    # 由于在得到NFA的状态转换图的时候，对于同一个项，有时候会产生重复的状态信息,例如上面的b就会重复创建两次有穷自动机
    data=[]
    data0 = [i for item in NFA_result for i in item]
    for i in data0:
        if i.isdigit():
            data.append(i)
    # print(self.NFA_result)
    data = list(set(data))
    # print(data)
    # NFA转DFA,得到DFA
    jk = NFA_DFA(NFA_result,start,end,data)
    re, last_end = jk.main0()
    print(re)
    la = MMFA(re,last_end)
    # DFA最小化
    mfa =  la.main1()
    print(mfa)
