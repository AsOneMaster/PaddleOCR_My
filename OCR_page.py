#学生：何迅
#创建时间：2022/7/12 15:27
# _*_ coding:utf-8 _*_
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from db.ocrService import OcrService
from math import ceil
import xlwt
"""启动数据库操作服务"""
ocr_service = OcrService()


class TableWidget(QWidget):
    control_signal = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(TableWidget, self).__init__(*args, **kwargs)
        self.__init_ui()

    def __init_ui(self):
        style_sheet = """
              QTableWidget {
                  border: none;
                  background-color:rgb(240,240,240)
              }
              QPushButton{
                  max-width: 18ex;
                  max-height: 6ex;
                  font-size: 11px;
              }
              QLineEdit{
                  max-width: 30px
              }
          """

        self.table = QTableWidget(5, 4)  # 5 行 4 列的表格
        self.table.setHorizontalHeaderLabels(["id", "角钢型号", "图片地址", "检测时间"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive | QHeaderView.Stretch)  # 自适应宽度
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Interactive | QHeaderView.Stretch)  # 自适应宽度
        self.__layout = QVBoxLayout()
        self.__layout.addWidget(self.table)
        self.setLayout(self.__layout)
        self.setStyleSheet(style_sheet)


    def setPageController(self, page):
        """自定义页码控制器"""
        control_layout = QHBoxLayout()
        homePage = QPushButton("首页")
        prePage = QPushButton("<上一页")
        self.curPage = QLabel("1")
        nextPage = QPushButton("下一页>")
        finalPage = QPushButton("尾页")
        self.totalPage = QLabel("共" + str(page) + "页")
        skipLable_0 = QLabel("跳到")
        self.skipPage = QLineEdit("1")
        skipLabel_1 = QLabel("页")
        confirmSkip = QPushButton("确定")
        self.update_sig = 1
        self.delete_sig = 1
        update = QPushButton("刷新")
        # self.export_sig = 1
        export_excel = QPushButton("导出")
        homePage.clicked.connect(self.__home_page)
        prePage.clicked.connect(self.__pre_page)
        nextPage.clicked.connect(self.__next_page)
        finalPage.clicked.connect(self.__final_page)
        confirmSkip.clicked.connect(self.__confirm_skip)
        update.clicked.connect(self.__update_view)
        export_excel.clicked.connect(self.export)
        # 更新表格数据，连接数据库
        self.table.itemChanged.connect(self.table_data_update)
        # 允许打开上下文菜单
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        # 绑定鼠标右键事件
        self.table.customContextMenuRequested.connect(self.generateMenu)
        control_layout.addStretch(1)
        control_layout.addWidget(homePage)
        control_layout.addWidget(prePage)
        control_layout.addWidget(self.curPage)
        control_layout.addWidget(nextPage)
        control_layout.addWidget(finalPage)
        control_layout.addWidget(self.totalPage)
        control_layout.addWidget(skipLable_0)
        control_layout.addWidget(self.skipPage)
        control_layout.addWidget(skipLabel_1)
        control_layout.addWidget(confirmSkip)
        control_layout.addWidget(update)
        control_layout.addWidget(export_excel)
        control_layout.addStretch(1)
        self.__layout.addLayout(control_layout)

    # 检测键盘回车按键，函数名字不要改，这是重写键盘事件
    # def keyPressEvent(self, event):
    #     # 这里event.key（）显示的是按键的编码
    #     # print("按下：" + str(event.key()))
    #     # # 举例，这里Qt.Key_A注意虽然字母大写，但按键事件对大小写不敏感
    #     # # if (event.key() == Qt.Key_Enter):
    #     # #     print('测试：Entq

    # 响应鼠标事件
    # def mousePressEvent(self, event):
    #     if event.button() == Qt.RightButton:
    #         print("鼠标右键点击")

    # Table 鼠标右键响应事件
    def generateMenu(self, pos):
        print(pos)

        # 获取点击行号
        for i in self.table.selectionModel().selection().indexes():
            column = i.column()
            rowNum = i.row()
        # 如果选择的行索引小于2，弹出上下文菜单
        if column == 1:
            menu = QMenu()
            item1 = menu.addAction("复制")
            item2 = menu.addAction("剪切")
            item3 = menu.addAction("粘贴")
            item4 = menu.addAction("删除")
            item5 = menu.addAction("修改")

            # 转换坐标系
            screenPos = self.table.mapToGlobal(pos)
            print("-----------------generateMenu鼠标右键函数:", screenPos)

            # 被阻塞
            action = menu.exec(screenPos)
            if action == item1:
                print('选择了第1个菜单项', self.table.item(rowNum, 0).text()
                      , self.table.item(rowNum, 1).text()
                      , self.table.item(rowNum, 2).text())
                text = self.table.item(rowNum, column).text()
                clipboard = QApplication.clipboard()
                clipboard.setText(text)
            elif action == item2:
                print('选择了第2个菜单项', self.table.item(rowNum, 0).text()
                      , self.table.item(rowNum, 1).text()
                      , self.table.item(rowNum, 2).text())
                text = self.table.item(rowNum, column).text()
                clipboard = QApplication.clipboard()
                clipboard.setText(text)
                self.table.setItem(rowNum, column, QTableWidgetItem(""))
            elif action == item3:
                # print('选择了第3个菜单项', self.table.item(rowNum, 0).text()
                #       , self.table.item(rowNum, 1).text()
                #       , self.table.item(rowNum, 2).text())
                clipboard = QApplication.clipboard()
                self.table.setItem(rowNum, column, QTableWidgetItem(clipboard.text()))
            elif action == item4:
                print('选择了第4个菜单项', self.table.item(rowNum, 0).text()
                      , self.table.item(rowNum, 1).text()
                      , self.table.item(rowNum, 2).text())
                identify = self.table.item(rowNum, 0).text()
                question = QMessageBox.question(self, '删除操作确认', '确定删除这一行吗？')
                if question == QMessageBox.Yes:
                    ocr_service.delete_row(identify)
                    # self.table.removeRow(rowNum)
                    self.control_signal.emit(["delete_sig", self.delete_sig])
                else:
                    return
            elif action == item5:
                print('选择了第5个菜单项', self.table.item(rowNum, 0).text()
                      , self.table.item(rowNum, 1).text()
                      , self.table.item(rowNum, 2).text())
                self.table.editItem(self.table.item(rowNum, column))
            else:
                return

    def __home_page(self):
        """点击首页信号"""
        self.control_signal.emit(["home", self.curPage.text()])
        # self.table_widget.show_data(int(self.curPage.text()) - 1)

    def __pre_page(self):
        """点击上一页信号"""
        self.control_signal.emit(["pre", self.curPage.text()])
        # self.table_widget.show_data(int(self.curPage.text()) - 1)

    def __next_page(self):
        """点击下一页信号"""
        self.control_signal.emit(["next", self.curPage.text()])


    def __final_page(self):
        """尾页点击信号"""
        self.control_signal.emit(["final", self.curPage.text()])
        # self.table_widget.show_data(int(self.curPage.text()) - 1)

    def __confirm_skip(self):
        """跳转页码确定"""
        self.control_signal.emit(["confirm", self.skipPage.text()])
        # self.table_widget.show_data(int(self.curPage.text()) - 1)

    def showTotalPage(self):
        """返回当前总页数"""
        return int(self.totalPage.text()[1:-1])

    def __update_view(self):
        """刷新页面"""
        self.control_signal.emit(["update_sig", self.update_sig])

    def table_data_update(self):
        """更新表格数据库数据"""
        data_select = self.table.selectedItems()
        # print("更新----------------------data_select:{}".format(data_select))
        if len(data_select) == 0:
            return
        row = self.table.currentRow()
        identify = self.table.item(row, 0).text()
        print("更新----------------------id:{}".format(identify))
        new_name = data_select[0].text()
        ocr_service.update_page(new_name, identify)
        # self.__update_view()

    def export(self):
        """导出excel表"""
        savefile_name = QFileDialog.getSaveFileName(self, '选择保存路径', '', 'Excel files(*.xls)')
        # print(savefile_name)
        if savefile_name[0]:
            path_savefile_name = savefile_name[0]
            book = xlwt.Workbook()
            sheet = book.add_sheet('新数据')
            result = ocr_service.select_all()
            row = len(result)
            col = len((result[0]))  # 去掉id
            content = ['检测结果', '图片地址', '记录时间']

            # for i in range(col):
            #     # self.tableWidget.horizontalHeaderItem(m).text()
            #     content.append('','')
            # print(content)
            for i in range(1):
                for j in range(col):
                    sheet.write(i, j, content[j])  # 去掉id
            for i in range(row):
                for j in range(col):
                    try:
                        sheet.write(i + 1, j, str(result[i][j]))  # 去掉id
                    except:
                        continue
                    # print(self.tableWidget.item(i,j).text())
            book.save(path_savefile_name)
            QMessageBox.information(self, "提示", "表格导出成功", QMessageBox.Yes)

    def show_data(self, page):
        """显示页面数据"""
        result = ocr_service.select_limit(page=page)
        for i in range(len(result)):
            for j in range(len(result[i])):
                self.newItem = QTableWidgetItem(format(result[i][j]))
                self.table.setItem(i, j, self.newItem)
                if j == 0:
                    self.table.setColumnHidden(j, True)    # 隐藏id列
                if j != 1:
                    self.newItem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

    def clear(self):
        """清除表格数据"""
        # self.table.setRowCount(0)
        self.table.clearContents()

class ChildWindow(QMainWindow):
    def __init__(self):
        super(ChildWindow, self).__init__()
        self.__init_ui()


    def __init_ui(self):
        self.resize(700, 300)
        self.setWindowTitle("角钢字符检测记录")
        self.table_widget = TableWidget()  # 实例化表格
        self.table_widget.setPageController(ceil(ocr_service.select_page()/5))  # 表格设置页码控制
        self.table_widget.show_data(0)  # 初始化首页数据
        self.table_widget.control_signal.connect(self.page_controller)  # 接送信号
        self.setCentralWidget(self.table_widget)

    # 页面变化
    def page_controller(self, signal):
        total_page = self.table_widget.showTotalPage()
        if "home" == signal[0]:
            index = self.table_widget.curPage.setText("1")
            # self.table_widget.show_data(int(index) - 1)
        elif "pre" == signal[0]:
            if 1 == int(signal[1]):
                QMessageBox.information(self, "提示", "已经是第一页了", QMessageBox.Yes)
                return
            self.table_widget.curPage.setText(str(int(signal[1]) - 1))
        elif "next" == signal[0]:
            if total_page == int(signal[1]):
                QMessageBox.information(self, "提示", "已经是最后一页了", QMessageBox.Yes)
                return
            self.table_widget.curPage.setText(str(int(signal[1]) + 1))
        elif "final" == signal[0]:
            self.table_widget.curPage.setText(str(total_page))
        elif "confirm" == signal[0]:
            if total_page < int(signal[1]) or int(signal[1]) < 0:
                QMessageBox.information(self, "提示", "跳转页码超出范围", QMessageBox.Yes)
                return
            self.table_widget.curPage.setText(signal[1])
        elif "update_sig" == signal[0]:
            QMessageBox.information(self, "提示", "刷新成功", QMessageBox.Yes)
        elif "delete_sig" == signal[0]:
            QMessageBox.information(self, "提示", "删除成功", QMessageBox.Yes)

        self.changeTableContent()

    # 改变表格内容
    def changeTableContent(self):
        """根据当前页改变表格的内容"""
        cur_page = self.table_widget.curPage.text()
        self.table_widget.clear()
        self.table_widget.totalPage = QLabel("共" + str(ceil(ocr_service.select_page()/5)) + "页")
        self.table_widget.show_data(int(cur_page) - 1)
        # pass




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChildWindow()
    window.show()
    sys.exit(app.exec_())
