from PyQt6 import QtWidgets, QtGui, QtCore

from src.gui.views.PanelView import PanelView
from src.libs.integration_v2 import compute_imr_for_cow
# from src.libs.serial_com import SerialCom

class PanelModel( QtWidgets.QWidget, PanelView ):
    def __init__(self, parent = None, *args ) -> None:
        super().__init__(parent, *args)

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

        self.BottomContent.AnalizeButton.setText("Analizando...")
        QtCore.QCoreApplication.processEvents()

        id_vaca = self.file_path.split("/")[-1].split(".")[0]
        resultado = compute_imr_for_cow(self.file_path, cow_id=id_vaca)

        self.BottomContent.CardProductividad.Metric.setText(str(round(resultado['merito_productivo'], 4)))
        self.BottomContent.CardProductividad.ZValue.setText(str(round(resultado['Z_merito'], 4)))

        self.BottomContent.CardComportamiento.Metric.setText(str(round(resultado['riesgo_comportamiento'], 4)))
        self.BottomContent.CardComportamiento.ZValue.setText(str(round(resultado['Z_riesgo_comport'], 4)))

        self.BottomContent.CardSalud.Metric.setText(str(round(resultado['riesgo_sanidad'], 4)))
        self.BottomContent.CardSalud.ZValue.setText(str(round(resultado['Z_riesgo_san'], 4)))

        self.BottomContent.IMR.setText( str( round( resultado['IMR'], 4 ) ) )

        if resultado['decision'] =='Retener / Reproducir':
            self.updateProperty( self.BottomContent.DecitionCard, "state", "Reproducir" )
            self.BottomContent.Decition.setText("Retener / Reproducir")
        
        elif resultado['decision'] =='Supervisar / Manejo dirigido':
            self.updateProperty( self.BottomContent.DecitionCard, "state", "Supervisar" )
            self.BottomContent.Decition.setText("Supervisar / Manejo dirigido")

        elif resultado['decision'] =='Descartar':
            self.updateProperty( self.BottomContent.DecitionCard, "state", "Descartar" )
            self.BottomContent.Decition.setText("Descartar")
        
        self.BottomContent.AnalizeButton.setText("Analizar datos")
        # {'id_vaca': 1221, 
        #  'merito_productivo': 1.1772926213360182, 
        #  'riesgo_comportamiento': 0.1689537107471228, 'riesgo_sanidad': 0.3896620276000007, 
        #  'Z_merito': 1.1772926213360182, 
        #  'Z_riesgo_comport': -3.310462892528772, 'Z_riesgo_san': -1.1033797239999932, 
        #  'IMR': 1.5817528063737616, 'decision': 'Retener / Reproducir'}