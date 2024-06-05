class MMFA:
    def __init__(self, re, last_end):
        self.dfa = re           # dfa
        self.final = last_end  # 终态
        self.element = []       # 存转换符
        self.all = []          # 存放没有遍历过的分组
        self.mfa = []          # 最终结果
        self.start = []         # 存储非终态

    # 找出所有转换符
    def trans_symbol(self):
        for li in self.dfa:
            if li[1] != '#':
                if li[1] not in self.element:
                    self.element.append(li[1])

    # 找出所有非终止符
    def find_start_state(self):
        for edge in self.dfa:
            if edge[0] not in self.final:
                if edge[0] not in self.start:
                    self.start.append(edge[0])
            if edge[2] not in self.final:
                if edge[2] not in self.start:
                    self.start.append(edge[2])

    # closure计算
    def jdg_one(self, i, a):
        for ll in self.dfa:
            if ll[0] == i:
                if ll[1] == a:
                    return ll[2]

    # 判断是否可分
    def judgement(self, tmp, i):
        # t = 0
        for a in self.element:  # 遍历每个转换符
            result = {}       # 用于存放起始点和终止点
            for t in tmp:     # 遍历每个起始点
                m = self.jdg_one(t, a)
                if m is None:
                    result[t] = -1
                else:
                    for j in range(0, len(self.all)):
                        if m in self.all[j]:
                            result[t] = j
            seen = []
            num = 0
            pp = []
            for key in result:
                temp0 = []
                if key not in seen:
                    for k in result:
                        if result[k] == result[key]:
                            num += 1
                            seen.append(k)
                            temp0.append(k)
                if len(temp0) > 0:
                    pp.append(temp0)
            # print(pp)
            if len(pp) > 1:
                # t = 1
                self.all.pop(i)
                for s in pp:
                    self.all.insert(i, s)
            # print(result)
            #     print(t)
                return 1
        return 0

    def main1(self):
        # 找出所有转换符
        self.trans_symbol()
        # 将所有终止符、非终止符加入列表中,做初步的划分
        self.all.append(self.final)
        self.find_start_state()
        self.all.append(self.start)
        # 遍历列表中所有的元素，长度为1的可以跳过
        i = 0    # 记录遍历位置，判断循环结束
        while 1:
            temp = self.all[i]
            if len(temp) == 1:  # 一个组内只有一个状态符的就不用再进行划分
                i += 1
            else:
                # print(temp)
                re = self.judgement(temp, i)
                # print(re)
                if re == 0:
                    i += 1
            if i >= len(self.all):
                break
        # print(self.all)
        # 等价替换
        for a in self.all:
            if len(a) > 1:
                p = a[1:]
                # print(p)
                for m in range (0, len(self.dfa)):
                    for n in self.dfa[m]:
                        if n in a:
                            replace = [a[0] if i in p else i for i in self.dfa[m]]
                            self.dfa.pop(m)
                            self.dfa.insert(m,replace)
        for b in self.dfa:
            if b not in self.mfa:
                self.mfa.append(b)
        return self.mfa