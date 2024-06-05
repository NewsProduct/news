# import numpy
class NFA_DFA:
    def __init__(self, NFA_result, start, end, data):
        self.nfa = NFA_result
        self.start = start
        self.end = end  # 用于查找终态
        self.end0 = []  # 用于查找终态存在的数组
        self.last_end = []  # 用于保存最后终态结果
        self.data = data
        self.element = []   # 用于记录nfa所有转换符
        self.closure = []    # 用于记录最后结果二维列表（实际是三维列表），第一列的内容
        self.closure0 = {}   # 用于记录所有元素到达下一个元素的步骤为空的集合
        self.alpha = []      # 用于记录move中会出现的所有结果ABC，即书上所说的将没有出现的状态集合转到第一列
        self.beta = []       # 用于记录还没遍历过的ABC
        self.list = []      # 用于存放最后的结果

    # 找出nfa中的转换符元素
    def trans_symbol(self):
        for l in self.nfa:
            if l[1] != '#':
                if l[1] not in self.element:
                    self.element.append(l[1])

    # 空字符处理,这里要进行深度优先的递归
    def closure_empty(self,start):
        c = []
        c.append(start)
        for l in self.nfa:
            if l[0] == start:
                if l[1] == '#':
                    c.append(l[2])
                    c.extend(self.closure_empty(l[2]))
        # 去重，因为其它状态可能会回到之前以及能到达的状态
        new_c = []
        for i in c:
            if i not in new_c:
                new_c.append(i)
        return new_c

    # 找所有元素的空字符到达
    def find_epsilon(self):
        for d in self.data:
            rest = self.closure_empty(d)
            if len(rest) > 0:
                self.closure0[d] = rest

    # 字符转换符处理
    # start是当前状态集合
    # trans_symbol是当前的转换符号
    def closure_letter(self, start, trans_symbol):
        c = []
        for i in start:
            # c.append(i)
            for l in self.nfa:
                if l[0] == i:
                    if l[1] == trans_symbol:
                        c.append(l[2])
                        # 此处的意思是，若0状态经过a到达1，同时1状态经过空到达2,3,此时需要将2,3加入到新状态集合
                        if l[2] in self.closure0.keys():
                            c.extend(list(self.closure0[l[2]]))
        # 由于start状态集合中的状态可能会达到相同的新状态，因此需要去重
        c = list(set(c))
        return c

    # 开始循环遍历整个nfa
    def dispose_nfa(self, A):
        # beta栈空则证明该遍历的ABC...都已经完成且没有新的产生
        a = 1
        while len(self.beta) > 0:
            a += 1
            # 每次将栈底元素推出栈
            aa = self.beta.pop(0)
            # 将元素添加进最终结果列表，因为是第一个所以需要提前加入
            self.closure.append(aa)
            # 遍历查询所有转换符
            # alpha和closure的区别是前者是存储了不为空的NFA到DFA的转换新状态集，但closure存了一些空是因为需要后续来生成新状态的符号
            for letter in self.element:
                # print(letter)
                res = self.closure_letter(aa, letter)
                if res not in self.alpha:
                    if len(res) > 0:
                        # print(res)
                        self.alpha.append(res)
                        self.beta.append(res)
                self.closure.append(res)
            # print(a)
            # print(self.closure)

    # 对最后结果进行规整，转成便于辨别的list形式
    def result_list(self, le):
        sign = ''
        a = 0
        for i in range(0, len(le)):
            # 主要是判断一个状态识别完所有的转移符号后，要换新的状态
            if (i + 1) % (len(self.element) + 1) == 1:
                sign = le[i]
                a = 0
            elif le[i] != '@':
                self.list.append([sign, self.element[a], le[i]])
                a += 1
            else:
                a += 1

    def main0(self):
        # 找出所有转换符
        self.trans_symbol()
        # 找所有元素的空字符到达
        self.find_epsilon()
        # 存储第一个closure为A，即初始状态经过空字符能到达的所有状态集合
        A = self.closure0[''.join(self.start)]
        # 将A存入已经出现过的list中
        self.alpha.append(A)
        # 将A存入到待遍历查找的list中
        self.beta.append(A)
        # 开始循环遍历整个nfa
        self.dispose_nfa(A)
        le = ['']*len(self.closure)
        al = 'A'
        length = 0
        con = []    # 记录已经改变过的列表
        for l in self.closure:
            if l not in self.end0:
                for aa in l:
                    if aa in self.end:
                        self.end0.append(l)
                        break
        for l in self.closure:
            t = 0
            # 如果不能构造，就直接跳出
            if length == len(self.closure):
                break
            elif l not in con:
                con.append(l)
                # 针对一些状态集合，其中的各个状态经过某符号之后，没有新状态的加入，即不能通过某符号转换到下一个状态
                if len(l) == 0:
                    for ii in range(0, len(self.closure)):
                        if self.closure[ii] == l:
                            length += 1
                            le[ii] = '@'
                else:
                    # 针对构造哪些可以输入运算符然后转换的状态集合
                    for ii in range (0, len(self.closure)):
                        if self.closure[ii] == l:
                            t = 1
                            length += 1
                            le[ii] = al
                            # 这里主要是处理终结符号集
                            if self.closure[ii] in self.end0:
                                if al not in self.last_end:
                                    self.last_end.append(al)
            # 这个表示上次的状态符号已经使用，我们需要新的状态符号了
            if t == 1:
                al = ord(al) + 1
                al = chr(al)
        self.result_list(le)
        return self.list, self.last_end

