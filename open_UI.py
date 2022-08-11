#学生：何迅
#创建时间：2021/12/17 15:06
import sys

from PyQt5.QtGui import QIcon

import OCR

from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':

    app = QApplication(sys.argv)

    MainWindow = QMainWindow()

    icon = QIcon("web/static/logo.png")

    MainWindow.setWindowIcon(icon)

    ui = OCR.Ui_MainWindow()

    ui.setupUi(MainWindow)

    MainWindow.show()

    sys.exit(app.exec_())