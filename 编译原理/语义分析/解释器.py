

from 编译原理.Analyse_cifa import Cifa
from 编译原理.语义分析.zhongjianAnalyse import zhongJianAnalyzer


class relolver:
    # lo为中间代码的对象
    def __init__(self,lo,gencode):
        self.lo = lo
        self.gencode = gencode  # 存储四元式
        self.output = []

    # 主要是解决负数问题,把负数字符串变为整数，然后弄到字典中去
    def initial(self,stack):
        for i in self.gencode:
            if i[1] != "":
                if i[1][0] == '-':
                    stack[i[1]] = -1*int(i[1][1:])
            if i[2] != "":
                if i[2][0] == '-':
                    stack[i[1]] = -1*int(i[1][1:])
        return stack

    def main1(self):
        stack = {}
        pc = 0
        # 用于生成write('\n')
        stack['\\n'] = ''
        stack = self.initial(stack)
        while True:
            i = self.gencode[pc]
            op = i[0]
            if op == 'main':
                pc = pc + 1
            elif op == '=':
                # isdigit 判断不了'-1'这种数字
                if str(i[1]).isdigit():
                    stack[i[3]] = int(i[1])
                else:
                    if i[1] == 'read' or i[1] == 'read()':
                        print("请输入一个数字：")
                        stack[i[3]] = int(input())
                    elif not str(i[1]).isdigit():
                        stack[i[3]] = stack[i[1]]
                    else:
                        stack[i[3]] = stack[i[1]] + stack[i[3]]
                pc = pc + 1
            elif op == '+':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = int(i[1]) + int(i[2])
                elif not i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = stack[i[1]] + int(i[2])
                elif i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = int(i[1]) + stack[i[2]]
                elif not i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = stack[i[1]] + stack[i[2]]
                pc = pc + 1
            elif op == '-':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = int(i[2]) - int(i[1])
                elif not i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = int(i[2]) - stack[i[1]]
                elif i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = stack[i[2]] - int(i[1])
                elif not i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = stack[i[2]] - stack[i[1]]
                pc = pc + 1
            elif op == '*':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = int(i[1]) * int(i[2])
                elif not i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = stack[i[1]] * int(i[2])
                elif i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = int(i[1]) * stack[i[2]]
                elif not i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = stack[i[1]] * stack[i[2]]
                pc = pc + 1
            elif op == '/':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = int(i[1]) / int(i[2])
                elif not i[1].isdigit() and i[2].isdigit():
                    stack[i[3]] = stack[i[1]] / int(i[2])
                elif i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = int(i[1]) / stack[i[2]]
                elif not i[1].isdigit() and not i[2].isdigit():
                    stack[i[3]] = stack[i[1]] / stack[i[2]]
                pc = pc + 1
            elif op == '%':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[2].isdigit() and i[1].isdigit():
                    stack[i[3]] = int(i[2]) % int(i[1])
                elif not i[2].isdigit() and i[1].isdigit():
                    stack[i[3]] = stack[i[2]] % int(i[1])
                elif i[2].isdigit() and not i[1].isdigit():
                    stack[i[3]] = int(i[2]) % stack[i[1]]
                elif not i[2].isdigit() and not i[1].isdigit():
                    stack[i[3]] = stack[i[2]] % stack[i[1]]
                pc = pc + 1
            elif op == '<':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[2].isdigit():
                    if stack[i[1]] < int(i[2]):
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
                else:
                    if stack[i[1]] < stack[i[2]]:
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
            elif op == '<=':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[2].isdigit():
                    if stack[i[1]] <= int(i[2]):
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
                else:
                    if stack[i[1]] <= stack[i[2]]:
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
            elif op == '>=':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[2].isdigit():
                    if stack[i[1]] >= int(i[2]):
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
                else:
                    if stack[i[1]] >= stack[i[2]]:
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
            elif op == '>':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[2].isdigit():
                    if stack[i[1]] > int(i[2]):
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
                else:
                    if stack[i[1]] > stack[i[2]]:
                        pc = int(i[3]) - 1
                    else:
                        pc = pc + 1
            elif op == '==':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if i[1].isdigit() and i[2].isdigit():
                    if int(i[1]) == int(i[2]):
                        pc = int(i[3]) - 1
                        continue
                elif not i[1].isdigit() and i[2].isdigit() :
                    if stack[i[1]] == int(i[2]):
                        pc = int(i[3]) - 1
                        continue
                elif i[1].isdigit() and not i[2].isdigit():
                    if int(i[1]) == stack[i[2]]:
                        pc = int(i[3]) - 1
                        continue
                elif not i[1].isdigit() and not i[2].isdigit() :
                    if stack[i[1]] == stack[i[2]]:
                        pc = int(i[3]) - 1
                        continue
                pc = pc + 1
            elif op == '!=':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if stack[i[1]] != int(i[2]):
                    pc = int(i[3]) - 1
                pc = pc + 1
            elif op == '!':
                pass
            elif op == '&&':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])
                if not i[1].isdigit() and not i[2].isdigit():
                    if stack[i[1]] != 0 and stack[i[2]] != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                if i[1].isdigit() and i[2].isdigit():
                    if int(i[1]) != 0 and int(i[2]) != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                if not i[1].isdigit() and i[2].isdigit():
                    if stack[i[1]] != 0 and int(i[2]) != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                if i[1].isdigit() and not i[2].isdigit():
                    if int(i[1]) != 0 and stack[i[2]] != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                pc = pc + 1
            elif op == '||':
                # 判断是否为全局变量
                if not i[1].isdigit() or not i[2].isdigit():
                    if i[1] not in stack and not i[1].isdigit():
                        stack[i[1]] = self.getValue(i[1])
                    if i[2] not in stack and not i[2].isdigit():
                        stack[i[2]] = self.getValue(i[2])

                if not i[1].isdigit() and not i[2].isdigit():
                    if stack[i[1]] != 0 or stack[i[2]] != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                if i[1].isdigit() and i[2].isdigit():
                    if int(i[1]) != 0 or int(i[2]) != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                if not i[1].isdigit() and i[2].isdigit():
                    if stack[i[1]] != 0 or int(i[2]) != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                if i[1].isdigit() and not i[2].isdigit():
                    if int(i[1]) != 0 or stack[i[2]] != 0:
                        stack[i[3]] = 1
                    else:
                        stack[i[3]] = 0
                pc = pc + 1
            elif op == 'j':
                if i[3] != '':
                    pc = int(i[3]) - 1
                elif i[3] == '':
                    pc = pc + 1
            elif op == 'para':
                pc = pc + 1
            elif op == 'call' and i[1] == 'write':
                s = self.gencode[pc - 1][1]
                if self.gencode[pc - 1][1] not in stack:
                    stack[self.gencode[pc - 1][1]] = self.getValue(self.gencode[pc - 1][1])
                if stack[self.gencode[pc - 1][1]] == None:
                    self.output.append(s)
                    print(s)
                else:
                    self.output.append(stack[self.gencode[pc - 1][1]])
                    print(stack[self.gencode[pc - 1][1]], end=" ")
                if i[3] != '':
                    pc = int(i[3]) - 1
                    continue
                else:
                    pc = pc + 1
            elif op == 'call' and i[1] == 'read':
                pc = pc + 1
            elif op == 'if':
                if i[3] != '':
                    pc = int(i[3]) - 1
                else:
                    pc = pc + 1
            elif op == 'else':
                pc = int(i[3]) - 1
            elif op == 'for':
                pc = int(i[3]) - 1
            elif op == 'while':
                pc = int(i[3]) - 1
            elif op == 'do-while':
                pc = int(i[3]) - 1
            elif op == 'break':
                pc = int(i[3]) - 1
            elif op == 'sys':
                break
        # print(stack)

    def getValue(self,ss):
        # 遍历变量符号表去找到对应的值
        for i in self.lo.bianliangBiaoOfYuyi:
            if i[1] == ss:
                return int(i[3])
        # 遍历常量符号表去找到对应的值
        for i in self.lo.changliangBiaoOfYuyi:
            if i[1] == ss:
                return int(i[3])


if __name__ == '__main__':
    f = open("./test/test11.txt", 'r', encoding='UTF-8-sig')
    words = f.read()+"     "
    # 词法分析
    Lex = Cifa(words)
    # 调用词法分析方法
    Lex.analy()
    # 得到token串
    LEX = Lex.CifaResult
    # 进行中间代码生成
    syntax_analyzer = zhongJianAnalyzer(LEX)
    syntax_tree = syntax_analyzer.parse()
    gencode = syntax_analyzer.gencode
    start = relolver(syntax_analyzer,gencode)
    # 解释器开始执行
    start.main1()