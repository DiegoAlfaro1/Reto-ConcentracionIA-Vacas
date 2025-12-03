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
        self.setCallbacks()   


# ============================= General Logic =============================

# ============ Reload qss dynamic properties ============

    def updateProperty( self, widget, property, value ):
            widget.setProperty(property, value)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

# ============ Asignar eventos a elementos de la interfaz ============

    def setCallbacks( self ):

        # Sensor
        # self.LoadButtonS.clicked.connect( self.ValidateConnectionSensor )
        self.BottomContent.CSVUploadButton.clicked.connect( self.UploadCSVFile )
        self.BottomContent.AnalizeButton.clicked.connect( self.ExecuteAnalysis )

    def UploadCSVFile( self ):
        self.file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv)")
        if self.file_path:
            print(f"Selected file: {self.file_path}")
            self.BottomContent.CSVUploadButton.setText(self.file_path)
            self.BottomContent.AnalizeButton.setEnabled(True)
            # return file_path
            
    def ExecuteAnalysis( self ):
        print("Executing data analysis...")