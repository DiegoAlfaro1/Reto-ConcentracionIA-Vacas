from PyQt6.QtWidgets import QApplication
from src.gui.WindowController import WindowController
from PyQt6 import QtGui

import ctypes

myappid = 'Concentrados.CowRank.subproduct.0.1.5' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

if __name__ == "__main__":
    app     = QApplication([])
    FontID = QtGui.QFontDatabase.addApplicationFont("src/resources/Inter-Variable.ttf")
    families = QtGui.QFontDatabase.applicationFontFamilies(FontID)
    if families:
        inter = families[0]
        app.setFont( QtGui.QFont(inter))
    # QtGui.QFontDatabase.addApplicationFont("src/resources/Montserrat-Regular.ttf")
    with open("src/styles/" + "styles.qss","r") as styleFile:
        app.setStyleSheet(styleFile.read())

    window = WindowController()
    window.show()
    app.exec()
