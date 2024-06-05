# 表达式转NFA
class NFA:
    def __init__(self, str):
        self.expression = str
        self.stack_start = []  # 用于存放开始结点
        self.stack_end = []  # 用于存放结束结点
        self.stack_re = [] # 用于存放结果
        self.num = 0

    def dispose_point(self):
        s = self.stack_start.pop(len(self.stack_start)-1)
        e = self.stack_end.pop(len(self.stack_end)-2)
        self.stack_re.append([e,'#',s])

    def dispose_star(self):
        s = self.stack_start.pop(len(self.stack_start) - 1)
        e = self.stack_end.pop(len(self.stack_end) - 1)
        self.stack_re.append([e,'#',s])
        self.stack_re.append([str(self.num),'#',s])
        self.stack_re.append([e,'#',str(self.num+1)])
        # 新前状态连接向新最后状态
        self.stack_re.append([str(self.num),'#',str(self.num+1)])
        self.stack_start.append(str(self.num))
        self.stack_end.append(str(self.num + 1))
        self.num += 2

    def dispose_or(self):
        s1 = self.stack_start.pop(len(self.stack_start) - 1)
        e1 = self.stack_end.pop(len(self.stack_end) - 1)
        s2 = self.stack_start.pop(len(self.stack_start) - 1)
        e2 = self.stack_end.pop(len(self.stack_end) - 1)
        # 即再构建两个新的状态，进行两条通路连接即可
        # 先创建终态连接是因为终态的状态数要默认大于初态
        self.stack_re.append([e1, '#', str(self.num+1)])
        self.stack_re.append([e2, '#', str(self.num + 1)])
        self.stack_re.append([str(self.num), '#', s1])
        self.stack_re.append([str(self.num), '#', s2])
        self.stack_start.append(str(self.num))
        self.stack_end.append(str(self.num + 1))
        self.num += 2


    # 主调用方法
    def express_NFA(self):
        for i in range(0,len(self.expression)):
            element = self.expression[i]
            # print(element)
            if element == '|':
                self.dispose_or()
            elif element == '*':
                self.dispose_star()
            elif element == '·':
                self.dispose_point()
            else:
                # 即此状态经过第一个字符输入后到达下一个状态
                self.stack_re.append([str(self.num),element,str(self.num+1)])
                self.stack_start.append(str(self.num))
                self.stack_end.append(str(self.num+1))
                self.num += 2
        return self.stack_re, self.stack_start, self.stack_end