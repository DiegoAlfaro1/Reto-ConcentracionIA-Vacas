# import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtGui import QCursor, QIcon

from random import randint

from src.gui.widgets.BarraSuperior import BarraSuperior
from src.gui.widgets.Contenido import Contenido
# from src.gui.widgets.Views.SensorView import *
# from src.gui.widgets.Views.ControlView import *
# from src.gui.widgets.Views.RutineView import *

# LIST OF COM PORTS : list_ports.comports():



#print(Units["Unit"][0])


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

        BottomContent = Contenido()

        MainVLayout.addWidget(TopBar)
        MainVLayout.addWidget(BottomContent)
        parent.setLayout( MainVLayout )
        


