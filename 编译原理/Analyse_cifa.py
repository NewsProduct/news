# 语法分析
# 种别码文件
import ast

from 编译原理.Lex import Lex
# 界符，运算符   这里需要注意单引号和双引号的识别也要进行转义处理
operateAndBlink = [' ','+', '*', '=', '<', '>', '&', '|', '!', '%', '^', '~',';', '{', '}', '(', ')', '[', ']', ',', ':','\n']
suangOperate = ['++','--','==','!=','>=','<=','&&','||','<<','>>','+=','-=','*=','/=','%=','&=','^=','|=']
sanOperate = ['<<=','>>=']
path = "./种别码.txt"
# 种别码字典

# 读取种别码
def getzhongb():
    zhongbie = {}
    with open(path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for s in content:
            clean_string = s.replace("\t", ' ').replace("\n", " ")
            zhongbie[clean_string.split(" ")[0]] = clean_string.split(" ")[1]
    return zhongbie

class Cifa:

    # 待处理源程序下标
    index = 0
    # 行号
    line = 0
    # token串
    CifaResult = []
    # 输入的源程序
    program = ""
    # 种别码
    zhongbie = {}
    # 错误信息处理
    erroInfo = []
    erroStr = ""

    def __init__(self,program):
        self.program = program
        self.zhongbie = getzhongb()


    # 分析函数
    def analy(self):
        while self.index<len(self.program):
            #    关键词和标识符处理
            if self.program[self.index].isalpha() or self.program[self.index] == "_" and self.index<len(self.program):
                self.processForFlagAndKeyWord(1)
            #     数值处理
            elif self.program[self.index].isdigit() and self.index<len(self.program):
                self.processForVarityNumber(2)
            #     注释处理
            elif self.program[self.index] == "/":
                self.proceeForexegesis(3)
            elif self.index < len(self.program) and self.program[self.index] == '\n':
                # 增加行数
                self.line+=1
            elif self.program[self.index] in operateAndBlink:
                # 界符处理
                self.processForBlink(4)
            # 其它符号的处理，例如#$@等，直接报错
            elif self.program[self.index] in "@#$~·":
                self.erro(9,911)
            #  字符和字符串的判断
            elif self.program[self.index] == '\'' or self.program[self.index] == '\"':
                self.proceesForZifu(5)
            #   负数判断
            elif self.program[self.index] == '-' :
                # 这里的一系列判断主要为了判断是否为负数,即如果一个负号前面是标识符，即此是算数表达式,则不是负数
                if self.program[self.index+1].isdigit() and self.program[self.index+1]!='0' \
                        and len(self.CifaResult)!=0 and self.CifaResult[len(self.CifaResult)-1].code not in ["400","500"] :
                    self.index+=1
                    self.processForVarityNumber(2)
                    # 得到负号后面的数字串
                    kk = self.CifaResult.pop().word
                    lex = Lex(self.line + 1, '-'+kk, 500)
                    self.CifaResult.append(lex)
                else:
                    lex = Lex(self.line + 1, '-', 302)
                    self.CifaResult.append(lex)
            else:
                print(self.program[self.index])
            self.index+=1

    # 字符串或者字符串处理
    def proceesForZifu(self,tag):
        sorce = self.program
        start = self.index
        state = 0

        if self.index<len(self.program):
            if self.program[self.index] == '\'':
                self.index += 1
                state=2
                while self.program[self.index] != '\'':
                    self.index+=1
                    if self.index>=len(self.program):
                        state = 4
                        break

        if self.index<len(self.program):
            if self.program[self.index] == '\"':
                self.index += 1
                state=1
                while self.program[self.index] != '\"':
                    self.index+=1
                    if self.program[self.index] == '\n':
                        self.line+=1
                    if self.index>=len(self.program):
                        state = 4
                        break
        if state!=4:
            #  先存左边的符号，然后存字符串，最后存最右边的
            word = sorce[start:start+1]
            zhongbieMa = self.getZhonb(word, tag, 0)
            lex = Lex(self.line + 1, word, zhongbieMa)
            self.CifaResult.append(lex)
            # 字符串内容
            word = sorce[start+1:self.index]
            zhongbieMa = self.getZhonb(word, tag, state)
            lex = Lex(self.line + 1, word, zhongbieMa)
            self.CifaResult.append(lex)
            # 字符内容
            word = sorce[self.index:self.index+1]
            zhongbieMa = self.getZhonb(word, tag, 0)
            lex = Lex(self.line + 1, word, zhongbieMa)
            self.CifaResult.append(lex)
        else:
            self.erro(tag, state)


    # 界符处理
    def processForBlink(self,tag):
        sorce = self.program
        word = sorce[self.index:self.index+1]
        # 跳过空白符号
        if word==' ':
            return
        if word == '\n':
            self.index -=1
            return
        # 注意，必须先判断是否为三目运算符
        # 判断是否为三目运算符
        #  判断三目运算符
        if sorce[self.index:self.index+3] in sanOperate:
            k = sorce[self.index:self.index + 3]
            zhongbieMa = self.getZhonb(k, tag, 0)
            lex = Lex(self.line + 1, k, zhongbieMa)
            self.CifaResult.append(lex)
            # 因为多拿了两个运算符
            self.index+=2
        elif sorce[self.index:self.index+2] in suangOperate:
            k = sorce[self.index:self.index + 2]
            zhongbieMa = self.getZhonb(k, tag, 0)
            lex = Lex(self.line + 1, k, zhongbieMa)
            self.CifaResult.append(lex)
            # 因为多拿了一个运算符
            self.index+=1
        #     最后就为单目运算符
        else:
            zhongbieMa = self.getZhonb(word, tag, 0)
            lex = Lex(self.line+1, word, zhongbieMa)
            self.CifaResult.append(lex)



    # 注释处理
    def proceeForexegesis(self,tag):
        source = self.program
        start = self.index
        state = 0
        stateExpire = [2,3,7]
        while state not in stateExpire and self.index<len(self.program):
            if state==0:
                if source[self.index] == "/":
                    state = 1
                    self.index +=1
                    continue
            elif state == 1:
                if source[self.index] == "/":
                    #   表示单行注释
                    state = 3
                    #  移动源程序下标，直至遇到换行符或者文件结束符
                    while self.index<len(self.program) and source[self.index] != '\n':
                        self.index+=1
                    if(source[self.index] == '\n'):
                        self.line+=1
                    continue
                #     处理多行注释
                elif source[self.index] == '*':
                    state = 4
                    self.index +=1
                    continue
                #     移动源程序下标，直至遇到*/
                else:
                    # 表示为运算符
                    state = 2
                    continue
            elif state == 4:
                if source[self.index] == '*':
                    state = 5
                    self.index += 1
                    continue
                # # /*/这样的错误
                # elif source[self.index] == '/':
                #     # 表示多行注释错误
                #     state = 8
                #     continue
                else:
                    if(self.index=='\n'):
                        self.line +=1
                    self.index += 1
                    continue
            elif state == 5:
                if source[self.index] == '/':
                    # 表示多行注释
                    state = 7
                    continue
                # 可能会出现/*******/这种情况
                else:
                    # 跳回之前的状态四
                    state = 4
                    self.index += 1
                    continue

        #   只有当为运算符的时候，才加入token
        if state == 2:
            word = source[start:self.index]
            zhongbieMa = self.getZhonb(word,tag,state)
            lex = Lex(self.line+1,word,zhongbieMa)
            self.CifaResult.append(lex)
            self.index -=1
        # if state==8 or state==9:
        #     # 输出错误信息
        #     self.erro(tag,state)


    # 错误处理
    def erro(self,tag,state):
        # 处理注释不规范
        if tag==3:
            if state==8 or state==9:
                strs = "第"+str(self.line+1)+"行: 出现多行注释错误"
                self.erroInfo.append(strs)
        elif tag == 9:
            if state == 911:
                strs = "第"+str(self.line+1)+"行: 出现错误字符，查看标识符或者关键词是否写错"
                self.erroInfo.append(strs)
        # 处理数字错误
        elif tag == 2:
            if state == 4:
                strs = "第"+str(self.line+1)+"行: 出现数字错误，疑似八进制书写错误"
                self.erroInfo.append(strs)
            elif state == 7:
                strs = "第"+str(self.line+1)+"行: 出现数字错误，疑似十六进制书写错误"
                self.erroInfo.append(strs)
            elif state == 16:
                strs = "第" + str(self.line+1) + "行: 出现数字错误，请你检查是否浮点数或者指数，科学计数法等有误"
                self.erroInfo.append(strs)
        elif tag == 1:
            if state==3:
                strs = "第"+str(self.line+1)+"行: 出现错误字符，查看标识符或者关键词是否写错"
                self.erroInfo.append(strs)
        elif tag == 5:
            if state == 4:
                strs = "第"+str(self.line+1)+"行: 出现错误字符，缺失\"或者\'"
                self.erroInfo.append(strs)



    # tag == 2表示这是识别整数的
    def processForVarityNumber(self,tag):
        hex_characters = ['a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F']
        source = self.program
        start = self.index
        state = 0
        # 终结状态
        stateExpire = [2,4,7,13,14,15,16,17]
        while state not in stateExpire and self.index < len(self.program):
            if state==0:
                if source[self.index] in "123456789":
                    state = 1
                    self.index+=1
                    continue
                if source[self.index] == '0':
                    state = 3
                    self.index+=1
                    continue
                else:
                    state = 16
            if state == 1:
                if source[self.index] in "0123456789":
                    self.index+=1
                    continue
                elif source[self.index]== ".":
                    state = 8
                    self.index+=1
                    continue
                elif source[self.index] == "e" or source[self.index] == "E" :
                    state = 10
                    self.index+=1
                    continue
                # 判断整数，空白符，界符，和运算符
                elif source[self.index] in operateAndBlink or source[self.index] =='/' or source[self.index] == '-':
                    # 为整数
                    state = 15
                    continue
                #     表示出错
                else:
                    # 可能的情况为1@，1A,这表示整数出错
                    state = 16
                    continue

             # 判断是八进制还是十六进制
            if state == 3:
                if source[self.index] in "01234567":
                    self.index+=1
                    continue
                elif source[self.index] == '.':
                    state = 8
                    self.index +=1
                    continue
                elif source[self.index] == "x" or source[self.index] =="X" :
                    state = 5
                    self.index+=1
                    continue
                elif source[self.index] in operateAndBlink:
                    # 终态为2表示八进制
                    state = 2
                    continue
                # 表示可能八进制书写错误
                else:
                    state = 4
                    continue
            #  判断十六进制
            if state == 5:
                if source[self.index].isdigit() or source[self.index] in hex_characters:
                    state = 6
                    self.index += 1
                    continue

            if state == 6:
                if source[self.index].isdigit() or source[self.index] in hex_characters:
                    self.index += 1
                    continue
                elif source[self.index] in operateAndBlink:
                    # 表示16进制正常结束
                    state = 17
                    continue
                else:
                    # 疑似十六进制书写错误
                    state = 7
                    continue
            if state == 8:
                if source[self.index].isdigit():
                    state = 9
                    self.index +=1
                    continue
                else:
                    # 数字错误
                    state = 16
                    continue
            if state == 9:
                if source[self.index].isdigit():
                    self.index += 1
                    continue
                elif source[self.index] in operateAndBlink:
                    # 为小数
                    state = 14
                    continue
                elif source[self.index] in "eE":
                    state = 10
                    self.index +=1
                    continue
                else:
                    state = 16
                    continue
            if state == 10:
                if source[self.index] in "+-":
                    state = 11
                    self.index+= 1
                    continue
                elif  source[self.index].isdigit():
                    state = 12
                    self.index +=1
                    continue
                else:
                    state = 16
                    continue
            if state == 11:
                if source[self.index].isdigit():
                    state = 12
                    self.index +=1
                    continue
                else:
                    state = 16
                    continue
            if state == 12:
                if source[self.index].isdigit():
                    self.index += 1
                    continue
                elif source[self.index] in operateAndBlink:
                    # 为指数
                    state = 13
                    continue
                else:
                    state = 16
                    continue
        # 得到关键词

        word = source[start:self.index]
        zhongbieMa = self.getZhonb(word,tag,state)
        # 标记出错误的行
        if state == 16 or state==4 or state==7:
            zhongbieMa = 0
            # 加载错误信息
            self.erro(tag, state)
        lex = Lex(self.line+1, word, zhongbieMa)
        self.CifaResult.append(lex)
        self.index -=1

        # 识别关键词和标识符
    def processForFlagAndKeyWord(self,tag):
        source = self.program
        start = self.index
        state = 0
        while state != 2 and state!=3:
            if state == 0:
                if source[self.index].isalpha() or source[self.index] == '_':
                    state = 1
                    self.index += 1
                else:
                    return
            elif state == 1:
                if source[self.index].isalnum() or source[self.index] == '_':
                    self.index += 1
                # 如果以空白符或者界符等结束，表示此正常
                elif source[self.index] in operateAndBlink or source[self.index] == '/' or source[self.index]=='-':
                    state = 2
                    continue
                else:
                    state = 3
                    continue

        word = source[start:self.index]
        zhongbieMa = self.getZhonb(word,tag,state)
        lex = Lex(self.line+1, word, zhongbieMa)  # Assuming line and stores are defined
        self.CifaResult.append(lex)
        self.erro(tag,state)
        self.index -= 1


    # 得到种别码
    def getZhonb(self, word,tag,state):
        # 表示识别字符串或者标识符
        if tag == 1:
            if word in self.zhongbie:
                return self.zhongbie[word]
            else:
                if(tag==1):
                    return self.zhongbie["标识符"]
            return 1
        # 表示识别的为整数
        elif tag == 2:
            # 八进制数
            if state == 2:
                return self.zhongbie["八进制"]
            # 十六进制数
            elif state == 17:
                return self.zhongbie["十六进制"]
            # 判断指数
            elif state == 13:
                return self.zhongbie["指数"]
            # 浮点数
            elif state == 14:
                return self.zhongbie["浮点数"]
            # 整数
            elif state == 15:
                return self.zhongbie["整数"]
            elif state == 16:
                # 出错
                return 0
        elif tag == 3:
            if state == 2:
                # 返回运算符除号
                return self.zhongbie['/']
        elif tag == 4:
            # 输出界符
            return self.zhongbie[word]
        elif tag ==5:
            if state==0:
                return self.zhongbie[word]
            elif state == 1:
                return self.zhongbie["字符串常量"]
            elif state == 2:
                return self.zhongbie["字符常量"]


# pp = r"D:\桌面文件\作业\编译原理\实验一\标识符测试.txt"
# with open(pp, 'r', encoding="utf-8") as lk:
#     content = "-1    "
#
#     temp = Cifa(content)
#     temp.analy()
#     opp =""
#     output = "{:<10} {:<10} {:<10}".format("行号", "关键字", "种别码")
#     opp += output +'\n'
#     print(output)
#     for s in temp.CifaResult:
#         output = "{:<10} {:<10} {:<10}".format(s.line, s.word, s.code)
#         opp += output+'\n'
#     print(opp)
#     # 输出错误信息
#     print(f"共{len(temp.erroInfo)}条错误：")
#     for m in temp.erroInfo:
#         print(m)