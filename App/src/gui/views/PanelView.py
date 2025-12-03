from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QCursor, QIcon

from random import randint

from src.gui.widgets.TopBar import TopBar
from src.gui.widgets.Content import Content

class PanelView( object ):
    def __init__( self ) -> None:

        pass
    
# ==================== Render UI ====================

    def setupUI( self, parent : QtWidgets.QWidget ):

        aflag = QtCore.Qt.AlignmentFlag
        MainVLayout = QtWidgets.QVBoxLayout()
        MainVLayout.setContentsMargins(0, 0, 0, 0)
        MainVLayout.setSpacing(0)

        TopBarWidget = TopBar()

        self.BottomContent = Content()

        MainVLayout.addWidget(TopBarWidget)
        MainVLayout.addWidget(self.BottomContent)
        parent.setLayout( MainVLayout )
        
