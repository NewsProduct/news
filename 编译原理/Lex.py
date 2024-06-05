
# 词法分析的类
class Lex:
    # 行号
    line = 0
    # 词
    word = 0
    # 种别码
    code = 0
    def __init__(self, line, word, code):
        self.line = line
        self.word = word
        self.code = code
