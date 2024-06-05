import ply.lex as lex

# 定义词法分析器的词法单元
tokens = (
    'RESERVED_WORD',
    'IDENTIFIER',
    'NUMBER',
    'OPERATOR',
    'STRING',
    'LEFT_BRACKET',
    'RIGHT_BRACKET',
    'LEFT_BRACE',
    'RIGHT_BRACE',
    'COLON',
    'SEMICOLON',
    'ILLEGAL_CHAR',
)

# 定义正则表达式模式
reservedWord = r'(while|if|else|switch|case|int|main|using|namespace|std|printf)'
identifier = r'[a-zA-Z_][a-zA-Z0-9_]*'
number = r'\d+'
operator = r'(\+|-|\*|<=|<|==|=|>=|>|>>|<<)'
string = r'\"[^\"]*\"'
left_bracket = r'\('
right_bracket = r'\)'
left_brace = r'\{'
right_brace = r'\}'
colon = r':'
semicolon = r';'
illegal_char = r'.'

# 定义规则
@lex.TOKEN(reservedWord)
def t_RESERVED_WORD(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(identifier)
def t_IDENTIFIER(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(number)
def t_NUMBER(t):
    t.value = (int(t.value), t.lineno, t.lexpos)
    return t

@lex.TOKEN(operator)
def t_OPERATOR(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(string)
def t_STRING(t):
    t.value = (t.value[1:-1], t.lineno, t.lexpos)
    return t

@lex.TOKEN(left_bracket)
def t_LEFT_BRACKET(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(right_bracket)
def t_RIGHT_BRACKET(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(left_brace)
def t_LEFT_BRACE(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(right_brace)
def t_RIGHT_BRACE(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(colon)
def t_COLON(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(semicolon)
def t_SEMICOLON(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

@lex.TOKEN(illegal_char)
def t_ILLEGAL_CHAR(t):
    t.value = (t.value, t.lineno, t.lexpos)
    return t

# 忽略空格和制表符
t_ignore = ' \t'

# 统计行号
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# 错误处理
def t_error(t):
    print("Illegal character '{}' at line {}, position {}".format(t.value[0], t.lineno, t.lexpos))
    t.lexer.skip(1)

# 构建词法分析器
lexer = lex.lex()

# 打开文件并读取内容
filename = "D:\桌面文件\作业\编译原理\实验一\标识符测试.txt"
with open(filename, 'r',encoding="utf-8") as file:
    data = file.read()

# 输入文本到词法分析器
lexer.input(data)

for tok in lexer:
    print(f'类型: {tok.type}, 内容: {tok.value}')