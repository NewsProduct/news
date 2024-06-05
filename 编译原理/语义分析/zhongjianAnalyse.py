from 编译原理.Analyse_cifa import Cifa
import queue

# 语法生成树
class TreeNode:
    def __init__(self, value):
        # 结点值
        self.value = value
        # 儿子结点
        self.children = []


class zhongJianAnalyzer:
    # 仅存储一个个词元
    tokens = []
    # 存储错误信息
    erro = []
    # 常量表
    changliangBiaoOfYuyi = []
    # 变量表
    bianliangBiaoOfYuyi = []
    # 函数表
    hanshuBiaoOfYuyi = []
    # 语句块栈,最外层为0
    zhan = "0"
    # 语句块深度统计
    sengdu = 0
    # 用于存储当前变量或常量类型
    style = ''
    # 存储四元式
    gencode = []
    # 存储语义错误
    yuyiEror = []
    # 是否是声明语句
    isShengm = 0
    # 算符优先栈
    resu = []
    r = 0
    # 函数调用位置
    positionHhanshuDY = 0
    #
    hanshuQueue = queue.Queue()
    # 用于确定continue的跳转位置
    continueStack = []

    def __init__(self, tokens):
        # 存储的是输入的tokens串对象，即lex对象
        self.backUpTokens = tokens
        self.current_token = None
        self.token_index = -1
        self.initialCiyuan()
        self.advance()

    # 从输入的tokens中拿到词元
    def initialCiyuan(self):
        for x in self.backUpTokens:
            self.tokens.append(x)

    # 获取词法单元的种别码
    # 有则返回正确的种别码，否则返回-1
    def judgeZhongbie(self):
        for x in self.backUpTokens:
            if self.current_token == x.word:
                return x.code
        return -1

    # 获取下一个词法单元
    def advance(self):
        self.token_index += 1
        if self.token_index < len(self.tokens):
            # 改
            self.current_token = self.tokens[self.token_index].word
        else:
            self.current_token = None

    # 匹配当前的词法单元是否和预期的一样，是则进入下一个词法单元，错误则报错
    def match(self, expected_token):
        if self.current_token and self.current_token == expected_token:
            # 更新作用域
            if expected_token == '{':
                self.sengdu +=1
                self.zhan += str(self.sengdu)
            elif expected_token == '}':
                # 出栈一个作用域范围
                self.zhan = self.zhan[0:len(self.zhan)-1]
            self.advance()
        else:
            strroo = f"第{self.tokens[self.token_index].line}行出现错误，预期的符号是:{expected_token},但输出的符号是:{self.current_token}"
            # raise SyntaxError(f"Syntax error: Expected '{expected_token}', found '{self.current_token}'")
            self.erro.append(strroo)

    # void pp int,int);
    # 递归入口
    def parse(self):
        return self.parse_prog()
    # 根节点的产生式: 程序 -> 声明语句 main () 复合语句 函数定义块
    def parse_prog(self):
        node = TreeNode("Prog")
        # 识别声明语句
        self.isShengm =1
        node.children.append(self.parse_decl())
        self.isShengm =0
        self.match("main")
        # 开始四元式
        self.gencode.append(['main','','',''])
        node.children.append(TreeNode("main"))
        self.match("(")
        node.children.append(TreeNode("("))
        self.match(")")
        node.children.append(TreeNode(")"))
        # 加入复合语句
        node.children.append(self.pase_fuheYuJu())
        # 结束四元式
        self.gencode.append(['sys','','',''])
        # 加入函数块
        node.children.append(self.HanShuKuai())
        return node


    # 函数块→函数定义 函数块|ε
    def HanShuKuai(self):
        node = TreeNode("HanShuKuai")
        if self.current_token in ["int", "void", "const"]:
            node.children.append(self.hanShuDingYi())
            node.children.append(self.HanShuKuai())
        else:
            return TreeNode("")
        return node

    # 函数定义→常量类型 变量 ( 函数定义形参列表 ) { 语句表 }|void 变量 ( 函数定义形参列表 ) { DD }
    def hanShuDingYi(self):
        node = TreeNode("hanShuDingYi")
        if self.current_token == "void":
            self.match("void")
            node.children.append(TreeNode("void"))
            if self.judgeZhongbie() == "400":
                # 生成函数定义四元式
                self.gencode.append([self.current_token,'','',''])
                node.children.append(TreeNode(self.current_token))
                self.advance()
            else:
                strerro = f"第{self.tokens[self.token_index].line}行出现错误，缺乏变量"
                self.erro.append(strerro)
            self.match("(")
            node.children.append(TreeNode("("))
            node.children.append(self.hanshuDingYiXinCanLieBiao())
            self.match(")")
            node.children.append(TreeNode(")"))
            self.match("{")
            node.children.append(TreeNode("{"))
            node.children.append(self.DD())
            self.match("}")
            node.children.append(TreeNode("}"))
        else:
            node.children.append(self.lexiChangl())
            if self.judgeZhongbie() == "400":
                # 生成函数定义四元式
                self.gencode.append([self.current_token,'','',''])
                node.children.append(TreeNode(self.current_token))
                self.advance()
            else:
                strerro = f"第{self.tokens[self.token_index].line}行出现错误，缺乏变量"
                self.erro.append(strerro)
            self.match("(")
            node.children.append(TreeNode("("))
            node.children.append(self.hanshuDingYiXinCanLieBiao())
            self.match(")")
            node.children.append(TreeNode(")"))
            self.match("{")
            node.children.append(TreeNode("{"))
            node.children.append(self.yuFaBiao())
            self.match("}")
            node.children.append(TreeNode("}"))
        return node


    # DD→const 常量类型 常量声明表 声明语句 EE|变量类型 N 声明语句 EE|void 变量 ( 函数声明形参列表 ) ; 声明语句 EE|执行语句 EE|ε
    def DD(self):
        node = TreeNode("DD")
        if self.current_token == "const":
            self.match("const")
            node.children.append(TreeNode("const"))
            node.children.append(self.lexiChangl())
            node.children.append(self.changLBiao())
            node.children.append(self.parse_decl())
            node.children.append(self.EE())
        elif self.current_token in ["int", "void"]:
            node.children.append(self.lexiBianl())
            node.children.append(self.N())
            node.children.append(self.parse_decl())
            node.children.append(self.EE())
        elif self.current_token == "void":
            self.match("void")
            node.children.append(TreeNode("void"))
            if self.judgeZhongbie() == "400":
                node.children.append(TreeNode(self.current_token))
                self.advance()
            else:
                strerro = f"第{self.tokens[self.token_index].line}行出现错误，缺乏变量"
                self.erro.append(strerro)
            self.match("(")
            node.children.append(TreeNode("("))
            node.children.append(self.hanshuLiebiao())
            self.match(")")
            node.children.append(TreeNode(")"))
            self.match(";")
            node.children.append(TreeNode(";"))
            node.children.append(self.parse_decl())
            node.children.append(self.EE())
        elif self.current_token in ["if","for","while","do","{"] or self.judgeZhongbie() == "400":
            node.children.append(self.zhiXinYuju())
            node.children.append(self.EE())
        else:
            return TreeNode("")
        return node


    # EE→DD|ε
    def EE(self):
        node = TreeNode("EE")
        node.children.append(self.DD())
        return node


    # 函数定义形参列表→函数定义形参|ε
    def hanshuDingYiXinCanLieBiao(self):
        node = TreeNode("hanshuDingYiXinCanLieBiao")
        # bl = self.tokens[self.token_index-3].word
        # num = self.token_index+1
        if self.current_token in ["int", "void", "const"]:
            node.children.append(self.hanShuDingYiXinCan())
            # num2 = self.token_index
            # temp = ''
            # for i in range(num, self.token_index):
            #     temp += self.tokens[i].word
            # self.gencode.append(['j'+bl, '', temp, len(self.gencode)+1])
        else:
            # self.gencode.append(['j' + bl, '', '', len(self.gencode) + 1])
            return TreeNode("")
        return node

    # 函数定义形参→变量类型 变量 FF
    def hanShuDingYiXinCan(self):
        node = TreeNode("hanShuDingYiXinCan")
        node.children.append(self.lexiBianl())
        if self.judgeZhongbie() == "400":
            node.children.append(TreeNode(self.current_token))
            self.advance()
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，缺乏变量"
            self.erro.append(strerro)
        node.children.append(self.FF())
        return node

    # FF→, 函数定义形参|ε
    def FF(self):
        node = TreeNode("FF")
        if self.current_token == ",":
            self.match(",")
            node.children.append(TreeNode(","))
            node.children.append(self.hanShuDingYiXinCan())
        else:
           return TreeNode("")
        return node


    # 这里是复合语句块
    # 复合语句 -> { 语句表 }
    def pase_fuheYuJu(self):
        node = TreeNode("FuheYuJu")
        self.match("{")
        node.children.append(TreeNode("{"))
        node.children.append(self.yuFaBiao())
        self.match("}")
        node.children.append(TreeNode("}"))
        return node

    # 语句表 -> const 常量类型 常量声明表 声明语句 X | 变量类型 N 声明语句 X | void 变量(函数声明形参列表) ; 声明语句 X | 变量 W X | Y X
    # | {语句表} X | 空
    def yuFaBiao(self):
        node = TreeNode("YuFaBiao")
        # 常量声明
        if self.current_token == "const":
            node.children.append(self.parse_const_decl())
            # 循环调用声明语句
            node.children.append(self.parse_decl())
            # 循环调用语句表
            node.children.append(self.X())
        elif self.current_token == 'void':
            # 进入函数声明
            node.children.append(self.pase_hanshu())
            # 又进入声明语句
            node.children.append(self.parse_decl())
            # 循环调用语句表
            node.children.append(self.X())
        elif self.current_token in ["int", "char", "float"]:
            # 进入变量声明
            node.children.append(self.parse_var_decl())
            # 又进入声明语句
            node.children.append(self.parse_decl())
            # 循环调用语句表
            node.children.append(self.X())
        elif self.judgeZhongbie() == "400":
            # 进入到变量 W X ,即可能性的函数调用
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.W())
            # 循环调用语句表
            node.children.append(self.X())
        elif self.current_token in ["if","for","while","do","return"]:
            # 进入执行语句
            node.children.append(self.Y())
            node.children.append(self.X())
        elif self.current_token == "{":
            node.children.append(self.yuFaBiao())
            self.match("}")
            node.children.append(self.X())
        elif self.current_token in ['break','continue']:
            # 这里要生成四元式
            if self.current_token == 'break':
                self.gencode.append(['break', '', '', ''])
            if self.current_token == 'continue':
                # 此处需要直接赋值
                self.gencode.append(['continue', '', '', ''])
            node.children.append(TreeNode(self.current_token))
            self.advance()
            self.match(";")
            node.children.append(TreeNode(";"))
            node.children.append(self.X())
        else:
            return TreeNode("")
        # 返回
        return node

    # X -> 语句表|空
    def X(self):
        node = TreeNode("X")
        node.children.append(self.yuFaBiao())
        return node

    # 已改
    # W-> = 表达式 ; | (实参列表) ;
    def W(self):
        node = TreeNode("W")
        bl = self.tokens[self.token_index-1].word
        num = self.token_index
        if self.current_token == "=":
            node.children.append(TreeNode("="))
            self.advance()
            num1 = self.token_index
            node.children.append(self.biaoDaShi())
            # 必须要放到表达式式子后面
            num2 = self.token_index
            # 就是为了判断后面可能是函数调用还是直接赋值,或者是一些复杂的表达式赋值
            if num2-num1 == 1:
                self.gencode.append(['=',self.tokens[self.token_index-1].word,'',bl])
                # 次数可能会产生变量赋值
                self.updateBianLiangBiao(self.tokens[self.token_index-1].word,bl)
            else:
                # 主要是为了得到形如read()样式的函数调用,或者后面为表达式
                s = ""
                for i in range(num+1,num2):
                    s += self.tokens[i].word
                self.gencode.append(['=',s,'',bl])
            # 请注意，在调用match函数后，只要没有出错，就是advance过的
            self.match(";")
            node.children.append(TreeNode(";"))
        # 这里一般用于处理函数调用
        elif self.current_token == "(":
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.shiCanLieBiao())
            # 如果此时是),代表参数para全部生成完毕，可以生成call四元式了
            if self.current_token == ')':
                self.gencode.append(['call',bl,'',''])
                # 此处是为了记录函数调用后的第一条四元式的位置
                self.hanshuQueue.put(len(self.gencode)+1)
            else:
                s = ''
                for i in range(num,self.token_index-1):
                    s += self.tokens[i].word
                self.gencode.append(['call',bl,'',s])
                # 此处是为了记录函数调用后的第一条四元式的位置
                self.hanshuQueue.put(len(self.gencode)+1)
            self.match(")")
            node.children.append(TreeNode(")"))
            self.match(";")
            node.children.append(TreeNode(";"))
        elif self.current_token in ["++","--"]:
            node.children.append(TreeNode(self.current_token))
            self.advance()
            self.match(";")
            node.children.append(TreeNode(";"))
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，预计是没有 '=' 或者 '('"
            self.erro.append(strerro)
        return node

    # 已改
    # 表达式 -> (算术表达式) B A C | 常量 B A C | 变量 D| ! 布尔表达式 E F
    def biaoDaShi(self):
        node = TreeNode("biaoDaShi")
        if self.current_token == "(" and self.tokens[self.token_index+1].word:
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.suanShuBiaoDaShi())
            self.match(")")
            node.children.append(TreeNode(")"))
            node.children.append(self.B())
            node.children.append(self.A())
            node.children.append(self.C())
        elif self.judgeZhongbie() =="400":
            # 判断变量
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.D())
        elif self.current_token.isdigit():
            # 判断常量
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.B())
            node.children.append(self.A())
            node.children.append(self.C())
        elif self.current_token =='\'' or self.current_token == '\"':
            # 判断字符常量,加入',"
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # 加入常量信息
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # 判断字符常量,加入',"
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.B())
            node.children.append(self.A())
            node.children.append(self.C())
        elif self.current_token == '!':
            # 判断布尔表达式
            node.children.append(TreeNode("!"))
            self.advance()
            self.gencode.append(['!', '', self.tokens[self.token_index - 1].word, len(self.gencode) + 2])
            node.children.append(self.buErBiaoDaShi())
            node.children.append(self.E())
            node.children.append(self.F())
        elif self.current_token == '-':
            node.children.append(TreeNode("-"))
            self.advance()
            node.children.append(self.biaoDaShi())
        elif self.current_token == ')':
            return TreeNode('')
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，表达式匹配错误"
            self.erro.append(strerro)
        return node

    # 已改
    # 算数表达式 -> 项 A
    def suanShuBiaoDaShi(self):
        node = TreeNode("suanShuBiaoDaShi")
        node.children.append(self.xiang())
        node.children.append(self.A())
        return node

    # 已改
    # 项 -> 因子 B
    def xiang(self):
        node = TreeNode("xiang")
        node.children.append(self.yingZi())
        node.children.append(self.B())
        return node
    # 已改
    # 因子 -> ( 算数表达式 ) | 常量 | 变量 G
    def yingZi(self):
        node = TreeNode("yingZi")
        if self.current_token == "(":
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.suanShuBiaoDaShi())
            self.match(")")
            node.children.append(TreeNode(")"))
            # 此处需要解释
            self.gencode.append(['=',len(self.gencode)+1,'',str(len(self.gencode)+2)])
        elif self.judgeZhongbie() =="400":
            # 判断变量
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.G())
        elif self.current_token.isdigit() or self.current_token[0] =='-':
            # 判断常量
            node.children.append(TreeNode(self.current_token))
            self.advance()
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，因子匹配错误"
            self.erro.append(strerro)
        return node

    # A -> + 算数表达式 | - 算数表达式 | 空
    def A(self):
        tmp = self.tokens[self.token_index-1].word
        node = TreeNode("A")
        s = ""
        if self.current_token == "+":
            node.children.append(TreeNode("+"))
            self.advance()
            num1 = self.token_index
            node.children.append(self.suanShuBiaoDaShi())
            num2 = self.token_index
            for i in range(num1,num2):
                s+=self.tokens[i].word
                if self.r >=2:
                    tmp = self.resu[self.r-2]
                    self.r -= 1
            # s = self.tokens[self.token_index - 1].word
            self.gencode.append(['+', s, tmp, tmp + '+' + s])
        elif self.current_token == "-":
            node.children.append(TreeNode("-"))
            self.advance()
            num1 = self.token_index
            # tmp = self.tokens[self.token_index - 3].word
            node.children.append(self.suanShuBiaoDaShi())
            num2 = self.token_index
            for i in range(num1, num2):
                s += self.tokens[i].word
                if self.r >= 2:
                    tmp = self.resu[self.r - 2]
                    self.r -= 1
            # s = self.tokens[self.token_index - 2].word
            self.gencode.append(['-', s, tmp, tmp + '-' + s])
        else:
            return TreeNode("")
        return node

    # 已改
    # B -> * 项 | / 项 | % 项 | 空
    def B(self):
        node = TreeNode("B")
        # 得到运算符前面的token
        tmp = self.tokens[self.token_index-1].word
        s =""
        if self.current_token == "*":
            node.children.append(TreeNode("*"))
            self.advance()
            num1 = self.token_index
            # tmp = self.tokens[self.token_index-3].word
            node.children.append(self.xiang())
            num2 = self.token_index
            for i in range(num1,num2):
                s += self.tokens[i].word
            result = tmp +"*"+s
            # resu为一个栈，主要的作用是用于算符优先，即乘除先行
            self.resu.append(result)
            self.r = self.r+1
            self.gencode.append(['*',s,tmp,result])
            # temp = self.tokens[self.token_index-2].word
            # self.gencode.append(['*',temp,tmp,str(tmp+"*"+temp)])
        elif self.current_token == "/":
            node.children.append(TreeNode("/"))
            self.advance()
            # tmp = self.tokens[self.token_index-3].word
            num1 = self.token_index
            node.children.append(self.xiang())
            num2 = self.token_index
            for i in range(num1, num2):
                s += self.tokens[i].word
            result = tmp + '/' + s
            self.resu.append(result)
            self.r = self.r+1
            self.gencode.append(['/',s,tmp,result])
            # temp = self.tokens[self.token_index-2].word
            # self.gencode.append(['/',temp,tmp,str(tmp+"/"+temp)])
        elif self.current_token == "%":
            node.children.append(TreeNode("%"))
            self.advance()
            # tmp = self.tokens[self.token_index-3].word
            num1 = self.token_index
            node.children.append(self.xiang())
            num2 = self.token_index
            for i in range(num1, num2):
                s += self.tokens[i].word
            result = tmp + '%' + s
            self.resu.append(result)
            self.r = self.r+1
            self.gencode.append(['%',s,tmp,result])
            # temp = self.tokens[self.token_index-2].word
            # self.gencode.append(['%',temp,tmp,str(tmp+"%"+temp)])
        else:
            return TreeNode("")
        return node

    # C -> 关系运算符 项 A H | && 布尔项 F | || 布尔表达式 | 空
    def C(self):
        node = TreeNode("C")
        if self.current_token in ["<","<=",'>',">=","==",'!=']:
            # num1 = self.token_index-1
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # num2 = self.token_index-1
            # s = ''
            # for i in range(num1,num2):
            #     s += self.tokens[i].word
            # if self.resu:
            #     s = self.resu[0]
            #     self.resu = None
            # self.gencode.append([self.tokens[self.token_index - 1].word, s, self.tokens[self.token_index].word,
            #                      s + self.tokens[self.token_index - 1].word + self.tokens[self.token_index].word])
            num1 = self.token_index
            node.children.append(self.xiang())
            node.children.append(self.A())
            node.children.append(self.H())
            num2 = self.token_index
            if num2-num1 == 1:
                if self.tokens[num1-1].word == '==' and not str(self.gencode[len(self.gencode)-1][3]).isdigit() and str(self.gencode[len(self.gencode)-1][3])!= '' :
                    self.gencode.append(
                        [self.tokens[num1 - 1].word, self.gencode[len(self.gencode)-1][3], self.tokens[num1].word, ''])
                else:
                    self.gencode.append([self.tokens[num1-1].word,self.tokens[num1-2].word,self.tokens[num1].word,''])
            else:
                self.gencode.append([self.tokens[num1-1].word,self.tokens[num1-2].word,self.gencode[len(self.gencode)-1][3],''])
        elif self.current_token == '&&':
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # temp = self.gencode.pop()
            # temp.pop()
            # temp.append(len(self.gencode) + 3)
            # self.gencode.append(temp)
            # self.gencode.append(['if', '', '', ''])
            self.gencode.append(
                [self.tokens[self.token_index - 1].word, self.tokens[self.token_index - 2].word, self.tokens[self.token_index].word,
                 self.tokens[self.token_index - 2].word + '&&' + self.tokens[self.token_index].word])
            node.children.append(self.buErXiang())
            node.children.append(self.F())
        elif self.current_token == '||':
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # 加入四元式
            # temp = self.gencode.pop()
            # temp.pop()
            # temp.append('')
            # self.gencode.append(temp)
            # self.gencode.append(['if', '', '', len(self.gencode) + 2])
            self.gencode.append(
                [self.tokens[self.token_index - 1].word, self.tokens[self.token_index - 2].word, self.tokens[self.token_index].word,
                 self.tokens[self.token_index - 2].word + '||' + self.tokens[self.token_index].word])
            node.children.append(self.buErBiaoDaShi())
        else:
            return TreeNode("")
        return node


    # D -> G B A C | =表达式
    def D(self):
        node = TreeNode("D")
        bl = self.tokens[self.token_index - 1].word
        num1 = self.token_index
        if self.current_token == '=':
            node.children.append(TreeNode("="))
            self.advance()
            node.children.append(self.biaoDaShi())
            num2 = self.token_index
            if num2 - num1 == 1:
                self.gencode.append(['=', self.tokens[self.token_index - 2].word, '', bl])
            else:
                s = ''
                for i in range(num1 +1, num2):
                    s += self.tokens[i].word
                self.gencode.append(['=', s, '', bl])
        else:
            node.children.append(self.G())
            node.children.append(self.B())
            node.children.append(self.A())
            node.children.append(self.C())
        return node



    # H -> E F | 空
    def H(self):
        node = TreeNode("H")
        if self.current_token in ["&&","||"]:
            node.children.append(self.E())
            node.children.append(self.F())
        else:
            return TreeNode("")
        return node
    # E -> && 布尔表达式
    def E(self):
        node = TreeNode("E")
        if self.current_token == "&&":
            temp = self.gencode.pop()
            temp.pop()
            temp.append(len(self.gencode) + 3)
            self.gencode.append(temp)
            self.gencode.append(['if', '', '', ''])
            node.children.append(TreeNode("&&"))
            self.advance()
            node.children.append(self.buErBiaoDaShi())
        else:
            return TreeNode("")
        return node

    # F -> || 布尔表达式
    def F(self):
        node = TreeNode("F")
        if self.current_token == "||":
            node.children.append(TreeNode("||"))
            self.advance()
            # 加入四元式
            temp = self.gencode.pop()
            temp.pop()
            temp.append(len(self.gencode) + 3)
            self.gencode.append(temp)
            self.gencode.append(['', '', '', ''])
            temp = self.gencode.pop()
            temp.pop()
            temp.append(len(self.gencode) + 3)
            self.gencode.append(temp)
            i = len(self.gencode)
            node.children.append(self.buErBiaoDaShi())
            self.gencode.append(['', '', '', ''])
            temp = self.gencode.pop()
            temp.pop()
            temp.append(len(self.gencode) + 3)
            self.gencode.insert(i, temp)
        else:
            return TreeNode("")
        return node

    # 已改
    # 布尔项 -> 布尔因子 E
    def buErXiang(self):
        node = TreeNode("buErXiang")
        node.children.append(self.buErYingZi())
        node.children.append(self.E())
        return node

    # 已改
    # 布尔因子 -> 表达式
    def buErYingZi(self):
        node = TreeNode("buErYingZi")
        node.children.append(self.biaoDaShi())
        return node

    # 已改
    # 布尔表达式 -> 布尔项 F
    def buErBiaoDaShi(self):
        node = TreeNode("buErBiaoDaShi")
        node.children.append(self.buErXiang())
        node.children.append(self.F())
        return node

    # G -> (实参列表) | 空
    def G(self):
        node = TreeNode("G")
        bl = self.tokens[self.token_index-1].word
        num = self.token_index
        if self.current_token == "(":
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.shiCanLieBiao())
            if self.tokens[self.token_index].word != ')':
                string = ''
                for i in range(num, self.token_index-1):
                    string += self.tokens[self.token_index].word
                self.gencode.append([bl, '', string, ''])
            # 为)代表被调用函数的参数para四元式已经生成，可以生成call四元式了
            else:
                self.gencode.append(['call', bl, '', ''])
                self.hanshuQueue.put(len(self.gencode)+1)
            self.match(")")
            node.children.append(TreeNode(")"))
        # 测试用例从未用过i++这段代码
        elif self.current_token == "++":
            node.children.append(TreeNode("++"))
            self.advance()
        elif self.current_token == "--":
            node.children.append(TreeNode("--"))
            self.advance()
        else:
            return TreeNode("")
        return node


    # 实参列表 -> 实参 | 空
    def shiCanLieBiao(self):
        node = TreeNode("shiCanLieBiao")
        node.children.append(self.shiCan())
        # 此处有疑问
        return node

    # 实参 -> 表达式 K
    def shiCan(self):
        node = TreeNode("shiCan")
        num1 = self.token_index
        node.children.append(self.biaoDaShi())
        num2 = self.token_index
        node.children.append(self.K())
        # 为什么四元式的生成这里放到最后的原因是函数调用，后面的参数要先生成
        # 此式表示参数是write(a,b)这种变量型的
        # 表示wirte('sedas')类型的,这里必须返回，不然下面也会生成四元式
        if self.tokens[num1].word =='\'' or self.tokens[num1] == '\"':
            self.gencode.append(['para',self.tokens[num1+1].word,'',''])
            return node
        if num2-num1 == 1:
            self.gencode.append(['para',self.tokens[num1].word,'',''])
        elif num2 == num1:
            # 这种情况用于应对调用函数没有参数
            pass
        else:
            # 拿到最新的四元式，即参数可能是复合语句
            mk = self.gencode[len(self.gencode)-1][3]
            self.gencode.append(['para',mk,'',''])
        return node

    # K -> , 实参 | 空
    def K(self):
        node = TreeNode("K")
        if self.current_token == ",":
            node.children.append(TreeNode(","))
            self.advance()
            node.children.append(self.shiCan())
        else:
            return TreeNode("")
        return node


    # 下面进入到执行类语句
    # Y -> 控制语句|return语句
    def Y(self):
        node = TreeNode("Y")
        if self.current_token in ["if", "for", "while", "do"]:
            node.children.append(self.kongZhiYuJu())
        elif self.current_token == "return":
            node.children.append(self.reTurnYuJu())
        return node

    # return语句→return CC
    def reTurnYuJu(self):
        node = TreeNode("reTurnYuJu")
        self.match("return")
        node.children.append(TreeNode("return"))
        node.children.append(self.CC())
        return node
    # CC→;|表达式 ;
    def CC(self):
        node = TreeNode("CC")
        if self.current_token == ";":
            # 生成返回值的四元式
            self.gencode.append(['ret','','',''])
            self.match(";")
            node.children.append(TreeNode(";"))
        else:
            num1 = self.token_index
            node.children.append(self.biaoDaShi())
            num2 = self.token_index
            if num2-num1 == 1:
                self.gencode.append(['ret', self.tokens[num1].word, '', ''])
            # 生成返回值的四元式,此处有具体的返回值
            else:
                self.gencode.append(['ret',self.gencode[len(self.gencode)-1][3],'',''])
            # self.gencode.append(['if','','',self.hanshuQueue.get()])
            self.match(";")
            node.children.append(TreeNode(";"))
        return node

    # 控制语句→if语句|for语句|while语句|do-while语句
    def kongZhiYuJu(self):
        node = TreeNode("kongZhiYuJu")
        if self.current_token == "if":
            node.children.append(self.if_stmt())
        elif self.current_token == "for":
            node.children.append(self.for_stmt())
        elif self.current_token == "while":
            node.children.append(self.while_stmt())
        elif self.current_token == "do":
            node.children.append(self.do_while_stmt())
        else:
            strroo = f"第{self.tokens[self.token_index].line}行出现错误，无效的控制语句"
            self.erro.append(strroo)
        return node

    # if语句→if ( 表达式 ) 语句 Z
    def if_stmt(self):
        node = TreeNode("if_stmt")
        self.match("if")
        node.children.append(TreeNode("if"))
        self.match("(")
        node.children.append(TreeNode("("))
        node.children.append(self.biaoDaShi())
        temp = self.gencode.pop()
        temp.pop()
        # 为什么要加三的原因是从它开始，还有一条if的假出口
        temp.append(len(self.gencode) + 3)
        # print(temp)
        self.gencode.append(temp)
        # if的假出口,加出口先空着
        self.gencode.append(['if', '', '', ''])
        self.match(")")
        node.children.append(TreeNode(")"))
        node.children.append(self.yuJu())
        node.children.append(self.Z())
        return node

    # Z→else 语句|ε
    def Z(self):
        node = TreeNode("Z")
        if self.current_token == "else":
            # else的加出口
            self.gencode.append(['else', '', '', ''])
            # 到生成else语句之后，应该回填if语句了，即if的假出口应该是else之后的第一条语句
            self.exit('if')
            self.match("else")
            node.children.append(TreeNode("else"))
            node.children.append(self.yuJu())
            self.exit('else')
            self.exit('if')
        else:
            # 没有else语句也要进行回填
            self.exit('if')
            # self.exit('j')
            return TreeNode("")
        return node

    # for语句 → for ( 表达式 ; 表达式 ; 表达式 ) 循环语句
    def for_stmt(self):
        node = TreeNode("for_stmt")
        self.match("for")
        node.children.append(TreeNode("for"))
        self.match("(")
        node.children.append(TreeNode("("))
        node.children.append(self.biaoDaShi())
        self.match(";")
        node.children.append(TreeNode(";"))
        # 真入口，即for语句的判断语句处，同时为了记录循环语句i++结束后的地方
        num = len(self.gencode)
        # 调用表达式函数
        node.children.append(self.biaoDaShi())
        temp = self.gencode.pop()
        temp.pop()
        # 加6是因为会生成假出口四元式，以及i++的
        temp.append(len(self.gencode) + 6)
        self.gencode.append(temp)
        # 假出口
        self.gencode.append(['for', '', '', ''])
        # 记录i++开始的四元式位置
        numss = len(self.gencode)
        self.match(";")
        node.children.append(TreeNode(";"))
        # 此处是为了便于continue找到循环语句的i++四元式所在位置
        self.continueStack.append({self.sengdu:len(self.gencode)+1})
        node.children.append(self.biaoDaShi())
        # i++完成之后，又回到判断语句
        self.gencode.append(['j','','',num+1])
        self.match(")")
        node.children.append(TreeNode(")"))
        # 加入循环语句
        node.children.append(self.xunHuanYuJu())
        # 循环语句块结束后就要进行无条件转移,返回到i++的地方
        self.gencode.append(['j','','',numss+1])
        self.exit('for')
        # 对于break，它的出口回填也是在整个for语句执行完成之后
        self.exit('break')
        return node

    # 出口回填
    def exit(self, s):
        # 因为是回填，所以从后向前遍历
        for i in range(len(self.gencode) - 1, -1, -1):
            if s == self.gencode[i][0]:
                # 即判断对应的四元式是否需要回填,前提是没有被回填过
                if type(self.gencode[i][3]) == str and len(self.gencode[i][3]) == 0:
                    temp = self.gencode.pop(i)
                    temp.pop()
                    # 这里加2而不是加一的原因是因为已经pop出了一个四元式
                    temp.append(len(self.gencode) + 2)
                    # 在完成回填之后，又加入到里面
                    self.gencode.insert(i, temp)
                    break


    # while语句→while ( 表达式 ) 循环语句
    def while_stmt(self):
        node = TreeNode("while_stmt")
        self.match("while")
        node.children.append(TreeNode("while"))
        self.match("(")
        node.children.append(TreeNode("("))
        # 记录它的原因是因为循环语句最后需要无条件返回判断处
        num = len(self.gencode)
        # 此处是为了便于continue找到循环语句的i++四元式所在位置
        self.continueStack.append({self.sengdu:len(self.gencode)+1})
        node.children.append(self.biaoDaShi())
        # 是将while里面的判断语句的真出口设置
        temp = self.gencode.pop()
        temp.pop()
        temp.append(len(self.gencode) + 3)
        self.gencode.append(temp)
        # while语句的假出口
        self.gencode.append(['while', '', '', ''])
        self.match(")")
        node.children.append(TreeNode(")"))
        node.children.append(self.xunHuanYuJu())
        self.gencode.append(['j','','',''])
        temp = self.gencode.pop()
        temp.pop()
        # 加一是因为num在记录的时候，表达式的四元式还未生成
        temp.append(num + 1)
        self.gencode.append(temp)
        self.exit('while')
        self.exit('break')
        return node

    # do-while语句→do 循环用复合语句 while ( 表达式 ) ;
    def do_while_stmt(self):
        node = TreeNode("do_while_stmt")
        self.match("do")
        node.children.append(TreeNode("do"))
        # 入口
        num = len(self.gencode)
        node.children.append(self.xunHuanFuheYuJu())
        self.match("while")
        node.children.append(TreeNode("while"))
        self.match("(")
        node.children.append(TreeNode("("))
        # 此处是为了便于continue找到循环语句的i++四元式所在位置
        self.continueStack.append({self.sengdu:len(self.gencode)+1})
        node.children.append(self.biaoDaShi())
        self.match(")")
        node.children.append(TreeNode(")"))
        self.match(";")
        node.children.append(TreeNode(";"))
        # 构建四元式
        temp = self.gencode.pop()
        temp.pop()
        temp.append(num + 1)
        self.gencode.append(temp)
        self.gencode.append(['do-while', '', '', len(self.gencode) + 2])
        self.exit('break')
        return node

    # 循环语句→语句
    def xunHuanYuJu(self):
        node = TreeNode("xunHuanYuJu")
        node.children.append(self.yuJu())
        return node

    # 语句→声明语句|执行语句
    def yuJu(self):
        node = TreeNode("yuJu")
        if self.current_token in ["const", "int", "char", "float", 'void']:
            # 加入声明语句
            node.children.append(self.parse_decl())
        elif self.current_token in ["break","continue"]:
            if self.current_token == 'break':
                self.gencode.append(['break','','',''])
            if self.current_token == 'continue':
                # 此处需要直接赋值
                self.gencode.append(['continue','','',''])
            node.children.append(TreeNode(self.current_token))
            self.advance()
        else:
            node.children.append(self.zhiXinYuju())
        return node

    # 执行语句→数据处理语句|控制语句|复合语句
    def zhiXinYuju(self):
        node = TreeNode("zhiXinYuju")
        if self.judgeZhongbie() == "400":
            node.children.append(self.shuJuChuLiYuJu())
        elif self.current_token in ["if", "for", "while", "do"]:
            node.children.append(self.kongZhiYuJu())
        elif self.current_token == "{":
            node.children.append(self.pase_fuheYuJu())
        else:
            strroo = f"第{self.tokens[self.token_index].line}行出现错误，无效的执行语句"
            self.erro.append(strroo)
        return node


    # 数据处理语句→变量 W
    def shuJuChuLiYuJu(self):
        node = TreeNode("shuJuChuLiYuJu")
        node.children.append(TreeNode(self.current_token))
        self.advance()
        node.children.append(self.W())
        return node

    # 循环用复合语句→{ 循环语句表 }
    def xunHuanFuheYuJu(self):
        node = TreeNode("xunHuanFuheYuJu")
        self.match("{")
        node.children.append(TreeNode("{"))
        node.children.append(self.xuHuanYuJuBiao())
        self.match("}")
        node.children.append(TreeNode("}"))
        return node
    # 循环语句表→循环语句 AA
    def xuHuanYuJuBiao(self):
        node = TreeNode("xuHuanYuJuBiao")
        node.children.append(self.xunHuanYuJu())
        node.children.append(self.AA())
        return node

    #   AA→循环语句表 | ε
    def AA(self):
        node = TreeNode("AA")
        if self.current_token in ["int", "void", "if", "for", "while", "do", "{",'break','continue'] or self.judgeZhongbie() == '400':
            node.children.append(self.xuHuanYuJuBiao())
        else:
            return TreeNode("")
        return node


    # 下面是声明语句
    # 声明语句→const 常量类型 常量声明表 声明语句|变量类型 N 声明语句|void 变量(函数声明形参列表);声明语句|ε
    # 函数声明 -> void 变量(函数声明形式参数列表);声明语句
    def parse_decl(self):
        node = TreeNode("Decl")
        # 判断是否结束main函数前的声明语句
        if self.current_token in ["const", "int", "char", "float", 'void']:
            # 常量声明
            if self.current_token == "const":
                node.children.append(self.parse_const_decl())
                # 循环调用声明语句
                node.children.append(self.parse_decl())
            else:
                # 判断是变量声明还是函数声明，直接进入main函数交给最上层的函数判断
                # 判断再后面两个的token是否为(
                if (self.current_token == 'void'):
                    self.style = 'void'
                    # 进入函数声明
                    node.children.append(self.pase_hanshu())
                    # 又进入声明语句
                    node.children.append(self.parse_decl())
                else:
                    # 进入变量声明
                    node.children.append(self.parse_var_decl())
                    # 又进入声明语句
                    node.children.append(self.parse_decl())
            # 返回
            return node

    # 函数声明
    # 函数声明 -> void 变量(函数声明形式参数列表);声明语句
    def pase_hanshu(self):
        node = TreeNode("pase_hanshu")
        if self.current_token == 'void':
            node.children.append(TreeNode("void"))
            self.advance()
            # 判断是否为变量
            if self.judgeZhongbie() == "400":
                node.children.append(TreeNode(self.current_token))
                self.advance()
                # match函数匹配成功后会自动调用advance
                self.match("(")
                node.children.append(TreeNode("("))
                # 调用函数声明形式参数列表
                node.children.append(self.hanshuLiebiao())
                self.match(")")
                node.children.append(TreeNode(")"))
                self.match(";")
                node.children.append(TreeNode(";"))
            else:
                strerro = f"第{self.tokens[self.token_index].line}行出现问题，无变量"
                self.erro.append(strerro)
        return node

    #  函数声明形式参数列表 -> 函数声明形参 |空
    def hanshuLiebiao(self):
        # 有可能没有形参，因此需要判空
        node = TreeNode("hanshuLiebiao")
        # bl = self.tokens[self.token_index-3].word
        # num = self.token_index - 1
        if self.current_token in ['int', 'char', 'float']:
            # 注意这里不能advance()
            node.children.append(self.hanShuXinC())
            # temp = ''
            # for i in range(num, self.token_index - 1):
            #     temp += self.tokens[i].word
            # self.gencode.append([bl, '', temp, len(self.gencode) + 1])
            return node
        else:
            # self.gencode.append([bl, '', '', len(self.gencode) + 1])
            return TreeNode("")

    # 函数声明形参 -> 变量类型 V
    def hanShuXinC(self):
        node = TreeNode("hanShuXinC")
        node.children.append(self.lexiBianl())
        node.children.append(self.V())
        return node

    # V -> ,函数声明形参 | 空
    def V(self):
        node = TreeNode("V")
        if self.current_token == ',':
            node.children.append(TreeNode(","))
            self.advance()
            node.children.append(self.hanShuXinC())
            return node
        else:
            return TreeNode("")

    # 已改
    # 变量声明
    # 变量声明 -> 变量类型 N(变量声明表) 声明语句
    def parse_var_decl(self):
        node = TreeNode("ValDecl")
        # 加入变量类型结点
        node.children.append(self.lexiBianl())
        # 加入变量声明表
        node.children.append(self.N())
        return node

    # 已改
    # 变量声明表
    # N->变量 O
    def N(self):
        node = TreeNode("BianlShengmBiao")
        # 查看此处是否为变量
        if self.judgeZhongbie() == "400":
            node.children.append(TreeNode(self.current_token))
            # 将此处的变量插入,函数内会检查是否变量重复命名
            self.insertbianLiangBiao()
            self.advance()
            # 进入下一个判段
            node.children.append(self.O())
        else:
            streerr = f"第{self.tokens[self.token_index].line}行出现问题，缺少变量"
            self.erro.append(streerr)
        return node

    # 已改
    # O -> PQ|(函数声明形参列表);
    # P -> = R|空
    # Q -> ;|, 变量声明表
    def O(self):
        node = TreeNode("O")
        # 判断是变量声明还是函数声明
        # 变量声明
        if self.current_token == '=':
            node.children.append(self.P())
            node.children.append(self.Q())
        #  函数声明,这种情况是对于非void声明的函数
        elif self.current_token == '(':
            # match函数匹配成功后会自动调用advance
            self.match("(")
            node.children.append(TreeNode("("))
            # 调用函数声明形式参数列表
            node.children.append(self.hanshuLiebiao())
            self.match(")")
            node.children.append(TreeNode(")"))
            self.match(";")
            node.children.append(TreeNode(";"))
        else:
            node.children.append(self.P())
            node.children.append(self.Q())
        return node

    # 已改
    # P -> =R | 空
    def P(self):
        # P
        node = TreeNode("P")
        if self.current_token == "=":
            # 加入子节点
            node.children.append(TreeNode("="))
            self.advance()
            node.children.append(self.R())
            return node
        else:
            return TreeNode("")

    # Q -> ;|,变量声明表
    def Q(self):
        node = TreeNode("Q")
        if self.current_token == ',':
            node.children.append(TreeNode(","))
            self.advance()
            node.children.append(self.bianlBiao())
        elif self.current_token == ';':
            node.children.append(TreeNode(";"))
            # 进入下一个判断
            self.advance()
        return node

    # 插入变量表
    def insertbianLiangBiao(self):
        flag =1
        if len(self.bianliangBiaoOfYuyi) == 0:
            self.bianliangBiaoOfYuyi.append([self.style, self.current_token, self.zhan])
            return
        for i in self.bianliangBiaoOfYuyi:
            # 同名变量在同一个作用域出现
            if i[1] == self.current_token and i[2] == self.zhan:
                strs = f"第{self.tokens[self.token_index].line}行出现语义错误，出现同名变量{self.current_token}"
                self.yuyiEror.append(strs)
                flag=0
        # 没有问题就插入
        if flag==1:
            self.bianliangBiaoOfYuyi.append([self.style, self.current_token, self.zhan])

    def updateBianLiangBiao(self,value,name):
        # 找打之前的变量信息
        for i in range(len(self.bianliangBiaoOfYuyi)-1,-1,-1):
            # 即精确查找到对应的变量
            if name == self.bianliangBiaoOfYuyi[i][1] and self.zhan == self.bianliangBiaoOfYuyi[i][2]:
                temp = self.bianliangBiaoOfYuyi.pop(i)
                # 加入属性值
                temp.append(value)
                self.bianliangBiaoOfYuyi.insert(i,temp)

    # 已改
    # R -> 常量 | 表达式
    def R(self):
        node = TreeNode("R")
        # 如果是常量
        if self.judgeZhongbie() == "400" or self.current_token.isdigit() or int(self.current_token)<0:
            node.children.append(TreeNode(self.current_token))
            # 加入四元式
            if self.isShengm == 0:
                self.gencode.append(['=',self.tokens[self.token_index].word,'',self.tokens[self.token_index-2].word])
            #不管是声明语句还是赋值语句，都要对变量的值进行插入
            self.updateBianLiangBiao(self.tokens[self.token_index].word,self.tokens[self.token_index-2].word)
            self.advance()
            if self.current_token == '(':
                node.children.append(TreeNode("("))
                self.advance()
                node.children.append(self.shiCanLieBiao())
                self.match(")")
                node.children.append(TreeNode(")"))
                self.match(";")
                node.children.append(TreeNode(";"))
        elif self.current_token == '\'' or self.current_token == '\"':
            node.children.append(TreeNode(self.current_token))
            # 加入四元式
            # self.gencode.append(['=',self.tokens[self.token_index+1].word,'',self.tokens[self.token_index-2].word])
            self.advance()
            # 此举是加入字符常量
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # 加入',"
            node.children.append(TreeNode(self.current_token))
            self.advance()
        else:
            # 加入表达式
            node.children.append(self.biaoDaShi())
        return node

    # 变量声明表 -> 单变量声明 Q
    def bianlBiao(self):
        node = TreeNode("bianlBiao")
        node.children.append(self.danbianl())
        node.children.append(self.Q())
        return node

    # 单变量
    # 单变量声明 -> 变量 U
    def danbianl(self):
        node = TreeNode("danbianl")
        # 判断我是否为标识符
        if self.judgeZhongbie() == "400":
            node.children.append(TreeNode(self.current_token))
            # 加入变量表
            self.bianliangBiaoOfYuyi.append([self.style, self.current_token, self.zhan])
            # 注意，正常情况下必须去找下一个次元
            self.advance()
            node.children.append(self.U())
        else:
            strreero = f"第{self.tokens[self.token_index].line}行出现错误，无变量"
            self.erro.append(strreero)
        return node
    # 已改
    # U -> =表达式 |空
    def U(self):
        # 需要判断为空
        node = TreeNode("U")
        if self.current_token == '=':
            # 这里需要生成四元式
            bl = self.tokens[self.token_index-1].word
            node.children.append(TreeNode("="))
            self.advance()
            num1 = self.token_index
            # 加入表达式
            node.children.append(self.biaoDaShi())
            num2 = self.token_index
            # 因为是表达式，所以是下一条四元式来解决
            if num2 - num1 ==1:
                if self.isShengm == 0:
                    self.gencode.append(['=',self.tokens[self.token_index-1].word,'',bl])
                    # 生成四元式的同时，更新符号表
                    self.updateBianLiangBiao(self.tokens[self.token_index-1].word,bl)
            else:
                if self.isShengm == 0:
                    self.gencode.append(['=',self.gencode[len(self.gencode)-1][3],'',bl])
            return node
        else:
            return TreeNode("")

    # 常量声明
    # 产生式：常量声明- > const 常量类型 常量声明表 声明语句
    def parse_const_decl(self):
        node = TreeNode("DeclOfChangl")
        node.children.append(TreeNode("const"))
        # 进入下一个token
        self.advance()
        # 进入常量类型判断
        node.children.append(self.lexiChangl())
        # 进入常量表
        node.children.append(self.changLBiao())
        return node

    # 已改
    # 常量声明表
    # 产生式子：常量声明表 -> 标识符 = S
    def changLBiao(self):
        node = TreeNode("changLBiao")
        # 判断是否为标识符
        if self.judgeZhongbie() == "400":
            # 在子树中加入标识符
            node.children.append(TreeNode(self.current_token))
            # 是标识符后，就加入常量表,此处可以直接加入值
            self.changliangBiaoOfYuyi.append([self.style,self.current_token,self.zhan,self.tokens[self.token_index+2].word])
            if self.isShengm == 0:
                self.gencode.append(['=',self.tokens[self.token_index+2].word,'',self.tokens[self.token_index].word])
            self.advance()
            # 匹配等号
            self.match("=")
            node.children.append(TreeNode("="))
            # 这里不能进入下一个token，因为不知道后面是否错误
            node.children.append(self.S())
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，缺少变量符号或者逗号多余"
            self.erro.append(strerro)
        return node

    # 已改
    # S-> 常量 T
    def S(self):
        node = TreeNode("S")
        # 判断是否为常量
        if self.current_token.isdigit() or self.judgeZhongbie() == "400":
            # 加入常量
            node.children.append(TreeNode(self.current_token))
            # 加入四元式,减2的原因是想找到等号左边的东西
            # self.gencode.append(['=',self.current_token,'',self.tokens[self.token_index-2].word])
            # 匹配下一个字符
            self.advance()
            node.children.append(self.T())
        #    进入此处代表遇到了字符变量或者字符串
        elif self.current_token == '\'' or self.current_token == '\"':
            # 加入常量
            node.children.append(TreeNode('\''))
            # 加入四元式,减2的原因是想找到等号左边的东西
            # self.gencode.append(['=',self.tokens[self.token_index+1].word,'',self.tokens[self.token_index-2].word])
            self.advance()
            # 加入字符或者字符串
            node.children.append(TreeNode(self.current_token))
            node.children.append(TreeNode(self.current_token))
            # 匹配 ',"
            self.match(self.current_token)
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，等号后面缺少常量或者变量"
            self.erro.append(strerro)
        return node

    # T -> ,|; 常量声明表
    def T(self):
        node = TreeNode("T")
        if self.current_token == ',':
            # 若后面还有则调用常量声明表
            node.children.append(TreeNode(','))
            # 已经可以正常进入下一个token判断了
            self.advance()
            node.children.append(self.changLBiao())
        elif self.current_token == ';':
            # 分号则代表没有了
            node.children.append(TreeNode(';'))
            # 正常进入下一个token判断
            self.advance()
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误,查看语句是否缺少分号或者逗号"
            self.erro.append(strerro)
        return node

    # 已改
    # 常量类型，即已经到了终结符的时候
    def lexiChangl(self):
        node = TreeNode("TypeChangl")
        if self.current_token in ['int', 'char', 'float']:
            # 设置当前变量（常量）的类型
            self.style = self.current_token
            node.children.append(TreeNode(self.current_token))
            # 这里进入下一个token表示没错
            self.advance()
            return node
        else:
            # 这里不进入下一个token，表示出现了错误，这里为了防止token的跳跃
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，没有定义常量类型"
            self.erro.append(strerro)
            return node

    # 已改
    # 变量类型，即已经到了终结符的时候
    def lexiBianl(self):
        node = TreeNode("TypeBianl")
        if self.current_token in ['int', 'char', 'float']:
            self.style = self.current_token
            node.children.append(TreeNode(self.current_token))
            # 这里进入下一个token表示没错
            self.advance()
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，没有定义变量类型"
            self.erro.append(strerro)
        return node

    # 函数类型，即已经到了终结符的时候
    def lexiHanshu(self):
        node = TreeNode("TypeHanshu")
        if self.current_token in ['int', 'char', 'float', 'void']:
            node.children.append(TreeNode(self.current_token))
            # 这里进入下一个token表示没错
            self.advance()
            return node
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，没有定义函数类型"
            self.erro.append(strerro)
            return node

    # 上面是声明语句

    # 打印函数
    def print_summary_view(self, node, indent=""):
        output = ""
        # 加上None判断比较好，防止打印报错，然后有助于排错
        if (node == None):
            return output

        if len(node.children) == 0:
            if node.value:
                # print(f"{indent} {node.value}")
                output += f"{indent} {node.value}\n"
        else:
            if node.value:
                # print(f"{indent} {node.value}")
                output += f"{indent} {node.value}\n"
            for child in node.children:
                # print_summary_view(child, indent + "   |")
                output += self.print_summary_view(child, indent + "  |")
        return output

    def getContinueTarget(self):
        for i in range(len(self.continueStack),-1,-1):
            zhanIn = self.continueStack[i].keys()[0]
            if self.sengdu - zhanIn >= 1:
                result = self.continueStack[i][zhanIn]
                self.continueStack.pop(i)
                return result


    def print_shiyuanshi(self):
        str = ""
        count = 1
        for i in self.gencode:
            # print(f"{count}:", end="")
            str += f"{count}:{i}\n"
            count += 1
        return str


if __name__ == '__main__':
    strs = """
    //双重for循环测试，求给定数以内的素数
    
    main(){
    
        int N = read() ;
        int count=0,nprime=0,i,j;
        for(i=2;i<=N;i=i+1) {
           nprime = 0;
           for(j=2;j<i;j=j+1) {
           if(i%j == 0) nprime = nprime + 1;
           }
           if(nprime == 0) {
                write(i);
                count = count + 1;
            }
         }
    
    }


       """
    Lex = Cifa(strs)
    Lex.analy()
    LEX = Lex.CifaResult
    syntax_analyzer = zhongJianAnalyzer(LEX)
    syntax_tree = syntax_analyzer.parse()
    for i in syntax_analyzer.erro:
        print(i)
    output = syntax_analyzer.print_summary_view(syntax_tree)
    # print(output)
    count = 1
    print("符号表:常量")
    for i in syntax_analyzer.changliangBiaoOfYuyi:
        print(i)
    print("符号表:变量")
    for i in syntax_analyzer.bianliangBiaoOfYuyi:
        print(i)
    for i in syntax_analyzer.gencode:
        print(f"{count}:",end="")
        print(i)
        count+=1

