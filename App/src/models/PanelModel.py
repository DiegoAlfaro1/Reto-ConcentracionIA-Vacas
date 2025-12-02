import time

from PyQt6 import QtWidgets, QtGui, QtCore

from src.gui.views.PanelView import PanelView
# from src.libs.serial_com import SerialCom

class PanelModel( QtWidgets.QWidget, PanelView ):
    def __init__(self, parent = None, *args ) -> None:
        super().__init__(parent, *args)
        super( QtWidgets.QWidget, self ).__init__()

        # Setup User Interface
        self.setupUI( self )
        # self.setCallbacks()


# ============================= General Logic =============================

# ============ Reload qss dynamic properties ============

    def updateProperty( self, widget, property, value ):
            widget.setProperty(property, value)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

# ============ Asignar eventos a elementos de la interfaz ============

    # def setCallbacks( self ):

        # Sensor
        # self.LoadButtonS.clicked.connect( self.ValidateConnectionSensor )
