
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QFileDialog, QScrollArea

from 编译原理.Analyse_cifa import Cifa
from 编译原理.正则表达式转换.DFA最小化 import MMFA
from 编译原理.正则表达式转换.NFA转DFA import NFA_DFA
from 编译原理.正则表达式转换.正则转NFA import zhengZetoNFA
from 编译原理.语义分析.zhongjianAnalyse import zhongJianAnalyzer
from 编译原理.语义分析.解释器 import relolver
from 编译原理.语法分析.yufaAnalyse import SyntaxAnalyzer


class ToolBarWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("工具栏")
        self.setGeometry(200, 200, 600, 400)

        # 创建主窗口中的中心部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建布局
        layout = QVBoxLayout(central_widget)

        # 创建工具栏布局
        toolbar_layout = QHBoxLayout()
        layout.addLayout(toolbar_layout)

        # 创建打开文件按钮
        open_button = QPushButton("文件(F)")
        toolbar_layout.addWidget(open_button)
        open_button.clicked.connect(self.open_file)

        # 创建编辑按钮
        edit_button = QPushButton("编辑(E)")
        toolbar_layout.addWidget(edit_button)
        edit_button.clicked.connect(self.edit_text)

        # 创建查看按钮
        view_button = QPushButton("查看(V)")
        toolbar_layout.addWidget(view_button)
        view_button.clicked.connect(self.view_text)
        # 创建处理按钮
        process_button = QPushButton("词法分析")
        toolbar_layout.addWidget(process_button)
        process_button.clicked.connect(self.cifaProcess_text)

        # 创建处理按钮
        process_button = QPushButton("语法分析")
        toolbar_layout.addWidget(process_button)
        process_button.clicked.connect(self.yufaProcess_text)

        # 创建处理按钮
        process_button = QPushButton("中间代码生成")
        toolbar_layout.addWidget(process_button)
        process_button.clicked.connect(self.zhongjianDaimaProcess_text)

        # 创建处理按钮
        process_button = QPushButton("解释器(控制台运行)")
        toolbar_layout.addWidget(process_button)
        process_button.clicked.connect(self.jieshiQiProcess_text)


        # 创建处理按钮
        process_button = QPushButton("正则式转化")
        toolbar_layout.addWidget(process_button)
        process_button.clicked.connect(self.zhengZeiProcess_text)




        # 创建内容区域布局
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)

        # 创建输入框区域
        input_area = QScrollArea()
        content_layout.addWidget(input_area)

        # 创建输入框
        self.input_text = QTextEdit()
        input_area.setWidget(self.input_text)
        input_area.setWidgetResizable(True)

        # 创建输出框区域
        output_area = QScrollArea()
        content_layout.addWidget(output_area)

        # 创建输出框
        # 第一输入框
        self.output_text1 = QTextEdit()
        # 错误输入框
        self.output_text2 = QTextEdit()

        output_layout = QVBoxLayout()
        output_layout.addWidget(self.output_text1)
        output_layout.addWidget(self.output_text2)

        output_widget = QWidget()
        output_widget.setLayout(output_layout)
        output_area.setWidget(output_widget)
        output_area.setWidgetResizable(True)

        self.resizeEvent = self.adjust_font_size  # 重写resizeEvent方法

    # 打开读取文件操作
    def open_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "打开文件")
        if file_path:
            with open(file_path, "r", encoding="utf-8") as file:
                self.input_text.setText(file.read())



    def zhengZeiProcess_text(self):
        try:
            # 清空输出框的文本内容
            self.output_text1.clear()
            self.output_text2.clear()
            # 输入文本
            input_text = self.input_text.toPlainText()
            mk = zhengZetoNFA(input_text)
            # 得到NFA
            NFA_result, start, end = mk.generate(input_text)
            s1 = 'NFA状态转换表：\n'
            s1 += "\t起点\t转换符\t终点\t\n"
            for member in NFA_result:
                s1 += '\t' + member[0] + '\t' + member[1] + '\t' + member[2] + '\t'+'\n'
            # 得到所有的状态
            data = []
            data0 = [i for item in NFA_result for i in item]
            for i in data0:
                if i.isdigit():
                    data.append(i)
            data = list(set(data))
            # print(data)
            # NFA转DFA,得到DFA
            jk = NFA_DFA(NFA_result, start, end, data)
            re, last_end = jk.main0()
            s2 = '\nDFA状态转换表：\n'
            s2 += "\t起点\t转换符\t终点\t\n"
            for member in re:
                s2 += '\t' + member[0] + '\t' + member[1] + '\t' + member[2] + '\t'+'\n'
            la = MMFA(re, last_end)
            # DFA最小化
            mfa = la.main1()
            s3 = '\nDFA最小化状态转换表：\n'
            s3 += "\t起点\t转换符\t终点\t\n"
            for member in re:
                s3 += '\t' + member[0] + '\t' + member[1] + '\t' + member[2] + '\t'+'\n'
            s = s1+s2+s3
            self.output_text1.setText(s)
        except Exception as e:
            # 捕获并处理异常
            error_message = f"运行出错: {str(e)}"
            self.output_text2.setPlainText(error_message)
            print(error_message)

    # 解释器前端显示
    def jieshiQiProcess_text(self):
        try:
            # 清空输出框的文本内容
            self.output_text1.clear()
            self.output_text2.clear()
            # 输入文本
            input_text = self.input_text.toPlainText() + "   "
            # 词法分析
            Lex = Cifa(input_text)
            # 调用词法分析方法
            Lex.analy()
            # 得到token串
            LEX = Lex.CifaResult
            # 进行中间代码生成
            syntax_analyzer = zhongJianAnalyzer(LEX)
            # 遍历语法树
            syntax_tree = syntax_analyzer.parse()
            # 提取四元式
            gencode = syntax_analyzer.gencode
            # 解释器运行
            start = relolver(syntax_analyzer, gencode)
            # 解释器开始执行
            start.main1()
            # 打印信息
            output = "程序输出结果为：\n"
            for item in start.output:
                if isinstance(item, str):
                    output += item+'\n'
                else:
                    output += str(item) +'\n'
            self.output_text1.setText(output)
            output = ''
            # # 清除掉之前的信息
            syntax_analyzer.erro = []
            syntax_analyzer.tokens = []
            # 清除
            syntax_analyzer.gencode = []
            # 清空分析对象的状态
            Lex.CifaResult = []
            Lex.erroInfo = []
            # 清空解释器信息
            start.lo= []
            start.gencode = []
        except Exception as e:
            # 捕获并处理异常
            error_message = f"运行出错: {str(e)}"
            # self.output_text2.setPlainText(error_message)
            # print(error_message)

    def printFuhaobiao(self,yufa):
        s1 = '常量表：\n'
        s1+= '\t类型\t名字\t作用域\t值\t\n'
        for i in yufa.changliangBiaoOfYuyi:
            s1 += '\t'+i[0]+'\t'+i[1]+'\t'+i[2]+'\t'+i[3]+'\t'+'\n'
        s2 = '变量表：\n'
        s2+= '\t类型\t名字\t作用域\t值\t\n'
        for i in yufa.bianliangBiaoOfYuyi:
            s2 += '\t'+i[0]+'\t'+i[1]+'\t'+i[2]+'\t'+i[3]+'\t'+'\n'
        s = s1+s2
        return s

    # 中间代码分析输出
    def zhongjianDaimaProcess_text(self):
        try:
            # 清空输出框的文本内容
            self.output_text1.clear()
            self.output_text2.clear()
            # 输入文本
            input_text = self.input_text.toPlainText() + "   "
            # 先进行词法分析
            Lex = Cifa(input_text)
            Lex.analy()
            # 判断是否有错误
            if (len(Lex.erroInfo) > 0):
                self.output_text2.setText("词法分析有错误，请先进行词法分析")
            else:
                # 得到token串
                LEX = Lex.CifaResult
                # 进行语法分析
                yufa = zhongJianAnalyzer(LEX)
                # 调用中间代码解析过程
                yufa.parse()
                # 判断是否有错误
                if (len(yufa.erro) > 0):
                    self.output_text2.setText("语法分析有错误，请先检查程序结构是否符合文法")
                # 打印四元式信息
                output = yufa.print_shiyuanshi()
                # 设置四元式生成
                self.output_text1.setText(output)
                fuhaobiao = self.printFuhaobiao(yufa)
                # 错误列表
                # erros = self.getErro(yufa.erro)
                self.output_text2.setText(fuhaobiao)
                # # 清除掉之前的信息
                yufa.erro = []
                yufa.tokens = []
                # 清除
                yufa.gencode.clear()
                # 清空分析对象的状态
                Lex.CifaResult = []
                Lex.erroInfo = []
        except Exception as e:
            # 捕获并处理异常
            error_message = f"运行出错: {str(e)}"
            # self.output_text2.setPlainText(error_message)
            # print(error_message)

    # 语法分析输出
    def yufaProcess_text(self):
        # 清空输出框的文本内容
        self.output_text1.clear()
        self.output_text2.clear()
        # 输入文本
        input_text = self.input_text.toPlainText() + "   "
        # 先进行词法分析
        Lex = Cifa(input_text)
        Lex.analy()
        # 判断是否有错误
        if(len(Lex.erroInfo) > 0):
            self.output_text2.setText("词法分析有错误，请先进行词法分析")
        else:
            # 得到token串
            LEX = Lex.CifaResult
            # 进行语法分析
            yufa = SyntaxAnalyzer(LEX)
            # 分析后得到语法树
            syntax_tree = yufa.parse()
            # 打印语法树
            output = yufa.print_summary_view(syntax_tree)
            # 设置语法树输出
            self.output_text1.setText(output)
            # 错误列表
            erros =self.getErro(yufa.erro)
            self.output_text2.setText(erros)
            # # 清除掉之前的信息
            yufa.erro.clear()
            yufa.tokens.clear()
            syntax_tree.children.clear()
            # 清空分析对象的状态
            Lex.CifaResult.clear()
            Lex.erroInfo.clear()

    # 词法分析输出
    def cifaProcess_text(self):
        # 清空输出框的文本内容
        self.output_text1.clear()
        self.output_text2.clear()

        input_text = self.input_text.toPlainText() + "   "
        # 此处进行词法分析
        temp = Cifa(input_text)
        temp.analy()
        opp = ""
        output = "{:<10} {:<10} {:<10}".format("行号", "关键字", "种别码")
        opp += output + '\n'
        for s in temp.CifaResult:
            output = "{:<10} {:<10} {:<10}".format(s.line, s.word, s.code)
            opp += output + '\n'
        self.output_text1.setText(opp)
        # 错误信息输出
        erro = self.getErro(temp.erroInfo)
        self.output_text2.setText(erro)

        # 清空分析对象的状态
        temp.CifaResult.clear()
        temp.erroInfo.clear()

    #     打印错误信息
    def getErro(self,erros):
        lens = len(erros)
        erroStr = "共"+str(lens)+"条错误:\n"
        for m in erros:
            erroStr += m+"\n"
        return erroStr

    # 共工具框调整
    def adjust_font_size(self, event):
        # 根据工具框的大小调整字体大小
        toolbar_size = self.centralWidget().size()
        font_size = min(toolbar_size.width(), toolbar_size.height()) // 70

        font = QFont()
        font.setPointSize(font_size)

        self.input_text.setFont(font)
        self.output_text1.setFont(font)
        self.output_text2.setFont(font)

    def edit_text(self):
        # 处理编辑按钮点击事件的逻辑
        # 可以在这里打开一个编辑对话框或执行其他编辑操作
        pass

    def view_text(self):
        # 处理查看按钮点击事件的逻辑
        # 可以在这里打开一个查看对话框或执行其他查看操作
        pass


if __name__ == "__main__":
    app = QApplication([])
    window = ToolBarWindow()
    window.show()
    app.exec_()