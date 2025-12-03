from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QCursor, QIcon

from random import randint

from src.gui.widgets.BarraSuperior import BarraSuperior
from src.gui.widgets.Contenido import Contenido

class PanelView( object ):
    def __init__( self ) -> None:

        pass
    
# ==================== Render UI ====================

    def setupUI( self, parent : QtWidgets.QWidget ):

        aflag = QtCore.Qt.AlignmentFlag
        MainVLayout = QtWidgets.QVBoxLayout()
        MainVLayout.setContentsMargins(0, 0, 0, 0)
        MainVLayout.setSpacing(0)

        TopBar = BarraSuperior()

        self.BottomContent = Contenido()

        MainVLayout.addWidget(TopBar)
        MainVLayout.addWidget(self.BottomContent)
        parent.setLayout( MainVLayout )
        
