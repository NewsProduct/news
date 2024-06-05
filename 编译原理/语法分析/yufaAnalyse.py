from 编译原理.Analyse_cifa import Cifa


# 语法生成树
class TreeNode:
    def __init__(self, value):
        # 结点值
        self.value = value
        # 儿子结点
        self.children = []


class SyntaxAnalyzer:
    # 仅存储一个个词元
    tokens = []
    # 存储错误信息
    erro = []

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
        node.children.append(self.parse_decl())
        self.match("main")
        node.children.append(TreeNode("main"))
        self.match("(")
        node.children.append(TreeNode("("))
        self.match(")")
        node.children.append(TreeNode(")"))
        # 加入复合语句
        node.children.append(self.pase_fuheYuJu())
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
        if self.current_token in ["int", "void", "const"]:
            node.children.append(self.hanShuDingYiXinCan())
        else:
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

    # W-> = 表达式 ; | (实参列表) ;
    def W(self):
        node = TreeNode("W")
        if self.current_token == "=":
            node.children.append(TreeNode("="))
            self.advance()
            node.children.append(self.biaoDaShi())
            self.match(";")
            node.children.append(TreeNode(";"))
        elif self.current_token == "(":
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.shiCanLieBiao())
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

    # 表达式 -> (算术表达式) B A C | 常量 B A C | 变量 D| ! 布尔表达式 E F
    def biaoDaShi(self):
        node = TreeNode("biaoDaShi")
        if self.current_token == "(" and self.tokens[self.token_index+1]:
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

    # 算数表达式 -> 项 A
    def suanShuBiaoDaShi(self):
        node = TreeNode("suanShuBiaoDaShi")
        node.children.append(self.xiang())
        node.children.append(self.A())
        return node

    # 项 -> 因子 B
    def xiang(self):
        node = TreeNode("xiang")
        node.children.append(self.yingZi())
        node.children.append(self.B())
        return node
    # 因子 -> ( 算数表达式 ) | 常量 | 变量 G
    def yingZi(self):
        node = TreeNode("yingZi")
        if self.current_token == "(":
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.suanShuBiaoDaShi())
            self.match(")")
            node.children.append(TreeNode(")"))
        elif self.judgeZhongbie() =="400":
            # 判断变量
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.D())
        # 这里多了一步判断负数
        elif self.current_token.isdigit() or self.current_token[0] == '-':
            # 判断常量
            node.children.append(TreeNode(self.current_token))
            self.advance()
        else:
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，因子匹配错误"
            self.erro.append(strerro)
        return node

    # A -> + 算数表达式 | - 算数表达式 | 空
    def A(self):
        node = TreeNode("A")
        if self.current_token == "+":
            node.children.append(TreeNode("+"))
            self.advance()
            node.children.append(self.suanShuBiaoDaShi())
        elif self.current_token == "-":
            node.children.append(TreeNode("-"))
            self.advance()
            node.children.append(self.suanShuBiaoDaShi())
        else:
            return TreeNode("")
        return node


    # B -> * 项 | / 项 | % 项 | 空
    def B(self):
        node = TreeNode("B")
        if self.current_token == "*":
            node.children.append(TreeNode("*"))
            self.advance()
            node.children.append(self.xiang())
        elif self.current_token == "/":
            node.children.append(TreeNode("/"))
            self.advance()
            node.children.append(self.xiang())
        elif self.current_token == "%":
            node.children.append(TreeNode("%"))
            self.advance()
            node.children.append(self.xiang())
        else:
            return TreeNode("")
        return node

    # C -> 关系运算符 项 A H | && 布尔项 F | || 布尔表达式 | 空
    def C(self):
        node = TreeNode("C")
        if self.current_token in ["<","<=",'>',">=","==",'!=']:
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.xiang())
            node.children.append(self.A())
            node.children.append(self.H())
        elif self.current_token == '&&':
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.buErXiang())
            node.children.append(self.F())
        elif self.current_token == '||':
            node.children.append(TreeNode(self.current_token))
            self.advance()
            node.children.append(self.buErBiaoDaShi())
        else:
            return TreeNode("")
        return node


    # D -> G B A C | =表达式
    def D(self):
        node = TreeNode("D")
        if self.current_token == '=':
            node.children.append(TreeNode("="))
            self.advance()
            node.children.append(self.biaoDaShi())
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
            node.children.append(self.buErBiaoDaShi())
        else:
            return TreeNode("")
        return node

    # 布尔项 -> 布尔因子 E
    def buErXiang(self):
        node = TreeNode("buErXiang")
        node.children.append(self.buErYingZi())
        node.children.append(self.E())
        return node
    # 布尔因子 -> 表达式
    def buErYingZi(self):
        node = TreeNode("buErYingZi")
        node.children.append(self.biaoDaShi())
        return node

    # 布尔表达式 -> 布尔项 F
    def buErBiaoDaShi(self):
        node = TreeNode("buErBiaoDaShi")
        node.children.append(self.buErXiang())
        node.children.append(self.F())
        return node

    # G -> (实参列表) | 空
    def G(self):
        node = TreeNode("G")
        if self.current_token == "(":
            node.children.append(TreeNode("("))
            self.advance()
            node.children.append(self.shiCanLieBiao())
            self.match(")")
            node.children.append(TreeNode(")"))
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
        node.children.append(self.biaoDaShi())
        node.children.append(self.K())
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
            self.match(";")
            node.children.append(TreeNode(";"))
        else:
            node.children.append(self.biaoDaShi())
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
        self.match(")")
        node.children.append(TreeNode(")"))
        node.children.append(self.yuJu())
        node.children.append(self.Z())
        return node

    # Z→else 语句|ε
    def Z(self):
        node = TreeNode("Z")
        if self.current_token == "else":
            self.match("else")
            node.children.append(TreeNode("else"))
            node.children.append(self.yuJu())
        else:
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
        node.children.append(self.biaoDaShi())
        self.match(";")
        node.children.append(TreeNode(";"))
        node.children.append(self.biaoDaShi())
        self.match(")")
        node.children.append(TreeNode(")"))
        # 加入循环语句
        node.children.append(self.xunHuanYuJu())
        return node

    # while语句→while ( 表达式 ) 循环语句
    def while_stmt(self):
        node = TreeNode("while_stmt")
        self.match("while")
        node.children.append(TreeNode("while"))
        self.match("(")
        node.children.append(TreeNode("("))
        node.children.append(self.biaoDaShi())
        self.match(")")
        node.children.append(TreeNode(")"))
        node.children.append(self.xunHuanYuJu())
        return node

    # do-while语句→do 循环用复合语句 while ( 表达式 ) ;
    def do_while_stmt(self):
        node = TreeNode("do_while_stmt")
        self.match("do")
        node.children.append(TreeNode("do"))
        node.children.append(self.xunHuanFuheYuJu())
        self.match("while")
        node.children.append(TreeNode("while"))
        self.match("(")
        node.children.append(TreeNode("("))
        node.children.append(self.biaoDaShi())
        self.match(")")
        node.children.append(TreeNode(")"))
        self.match(";")
        node.children.append(TreeNode(";"))
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
            node.children.append(TreeNode(self.current_token))
            self.advance()
            self.match(";")
            node.children.append(TreeNode(self.current_token))
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
        if self.current_token in ["int", "void", "if", "for", "while", "do", "{"]:
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
        if self.current_token in ['int', 'char', 'float']:
            # 注意这里不能advance()
            node.children.append(self.hanShuXinC())
            return node
        else:
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

    # 变量声明
    # 变量声明 -> 变量类型 N(变量声明表) 声明语句
    def parse_var_decl(self):
        node = TreeNode("ValDecl")
        # 加入变量类型结点
        node.children.append(self.lexiBianl())
        # 加入变量声明表
        node.children.append(self.N())
        return node

    # 变量声明表
    # N->变量 O
    def N(self):
        node = TreeNode("BianlShengmBiao")
        # 查看此处是否为变量
        if self.judgeZhongbie() == "400":
            node.children.append(TreeNode(self.current_token))
            self.advance()
            # 进入下一个判段
            node.children.append(self.O())
        else:
            streerr = f"第{self.tokens[self.token_index].line}行出现问题，缺少变量"
            self.erro.append(streerr)
        return node

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

    # R -> 常量 | 表达式
    def R(self):
        node = TreeNode("R")
        # 如果是常量
        if self.judgeZhongbie() == "400" or self.current_token.isdigit() or int(self.current_token)<0:
            node.children.append(TreeNode(self.current_token))
            self.advance()
            if self.current_token == '(':
                node.children.append(TreeNode("("))
                self.advance()
                node.children.append(self.shiCanLieBiao())
                self.match(")")
                node.children.append(TreeNode(")"))
                self.match(";")
                node.children.append(TreeNode(";"))
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
            # 注意，正常情况下必须去找下一个次元
            self.advance()
            node.children.append(self.U())
        else:
            strreero = f"第{self.tokens[self.token_index].line}行出现错误，无变量"
            self.erro.append(strreero)
        return node

    # U -> =表达式 |空
    def U(self):
        # 需要判断为空
        node = TreeNode("U")
        if self.current_token == '=':
            node.children.append(TreeNode("="))
            self.advance()
            # 加入表达式
            node.children.append(self.biaoDaShi())
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

    # 常量声明表
    # 产生式子：常量声明表 -> 标识符 = S
    def changLBiao(self):
        node = TreeNode("changLBiao")
        # 判断是否为标识符
        if self.judgeZhongbie() == "400":
            # 在子树中加入标识符
            node.children.append(TreeNode(self.current_token))
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

    # S-> 常量 T
    def S(self):
        node = TreeNode("S")
        # 判断是否为常量
        if self.current_token.isdigit() or self.judgeZhongbie() == "400":
            # 加入常量
            node.children.append(TreeNode(self.current_token))
            # 匹配下一个字符
            self.advance()
            node.children.append(self.T())
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

    # 常量类型，即已经到了终结符的时候
    def lexiChangl(self):
        node = TreeNode("TypeChangl")
        if self.current_token in ['int', 'char', 'float']:
            node.children.append(TreeNode(self.current_token))
            # 这里进入下一个token表示没错
            self.advance()
            return node
        else:
            # 这里不进入下一个token，表示出现了错误，这里为了防止token的跳跃
            strerro = f"第{self.tokens[self.token_index].line}行出现错误，没有定义常量类型"
            self.erro.append(strerro)
            return node

    # 变量类型，即已经到了终结符的时候
    def lexiBianl(self):
        node = TreeNode("TypeBianl")
        if self.current_token in ['int', 'char', 'float']:
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



# if __name__ == '__main__':
#     strs = """
# //算术表达式
# main() {
#        int x, y, z;
#        x = 9;
#        y = 3;
#        z = x - y * y - x % 3;
#        write(x);
#        write(y);
#        write(z);
#        }
#        """
#     Lex = Cifa(strs)
#     Lex.analy()
#     LEX = Lex.CifaResult
#     syntax_analyzer = SyntaxAnalyzer(LEX)
#     syntax_tree = syntax_analyzer.parse()
#     for i in syntax_analyzer.erro:
#         print(i)
#     output = syntax_analyzer.print_summary_view(syntax_tree)
#     print(output)
#
