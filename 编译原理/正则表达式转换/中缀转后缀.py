import sys


class ZhongZuiToHouZui:
    def __init__(self, str):
        self.str = str
        self.postfix = ""
        # 栈内优先级，栈顶的字符的优先级 d代表一切字母
        self.isp = {'·': 5, '|': 3, '(': 1, ')': 7, '*': 7, '#': 1, 'd': 7}
        # 栈外优先级，当前扫描到的字符的优先级
        self.icp = {'·': 4, '|': 2, '(': 6, ')': 1, '*': 6, '#': 1, 'd': 8}

    def ispFunc(self, char):
        priority = self.isp.get(char, -1)
        if priority == -1:
            print("error: 出现未知符号！")
            sys.exit(1)  # 异常退出
        return priority

    def icpFunc(self, char):
        priority = self.icp.get(char, -1)
        if priority == -1:
            print("error: 出现未知符号！")
            sys.exit(1)  # 异常退出
        return priority

    def in2post(self):
        str = self.str + '#'
        # 入栈结束符
        stack = ['#']
        loc = 0
        while stack and loc < len(str):
            c1, c2 = stack[-1], str[loc]
            # a = c1
            tmp = c2
            if c1.isalpha():
                c1 = 'd'
            if c2.isalpha():
                c2 = 'd'
            if self.ispFunc(c1) < self.icpFunc(c2):
                # 栈外字符优先级更高，压栈
                stack.append(tmp)
                loc += 1  # 前进
            elif self.ispFunc(c1) > self.icpFunc(c2):
                # 栈顶字符优先级更高，弹出
                self.postfix += stack.pop()
            else:
                # 优先级相等，要么结束了，要么碰到右括号了，都弹出但并不添加至后缀表达式中
                stack.pop()
                loc += 1

    def getResult(self):
        self.in2post()
        return self.postfix


if __name__ == '__main__':
    str = "c|(d*|a)·d·d"  # 简单测试
    # infix_expression = input("请输入算式表达式：")
    print("infix_expression:", str)
    solution = ZhongZuiToHouZui(str)
    print("postfix_expression:", solution.getResult())