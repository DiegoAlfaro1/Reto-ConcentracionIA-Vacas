from PyQt6 import QtWidgets, QtCore, QtGui
from src.gui.widgets.Carta import Carta
from src.gui.widgets.CartaMetricas import CartaMetricas


class Contenido( QtWidgets.QWidget ):
    def __init__( self, parent = None):
        super().__init__( parent )

        self.setupUI()

    def setupUI( self ):

        aflag = QtCore.Qt.AlignmentFlag
        

# ====================  Layout de contenido ====================

        ContentLayout = QtWidgets.QVBoxLayout()
        ContentLayout.setContentsMargins(0, 0, 0, 0)
        ContentLayout.setSpacing(0)
        self.setLayout(ContentLayout)
        self.setObjectName("Content-Background")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.MinimumExpanding)



# ==================== Layout para subir el CSV ====================

        CSVLayout = QtWidgets.QHBoxLayout()
        CSVLayout.setContentsMargins(25, 25, 25, 12)
        ContentLayout.addLayout( CSVLayout )

        CardCSV = Carta()
        CardCSV.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)

        CardCSVTitle = QtWidgets.QLabel( 
            "Subir registros CSV",
            objectName="Card-Title")
        CardCSVTitle.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        CardCSVTitle.setWordWrap(True) 
        CardCSV.addWidget(  CardCSVTitle )

        CSVUploadLayout = QtWidgets.QHBoxLayout()
        CardCSV.addLayout( CSVUploadLayout )

        self.CSVUploadButton = QtWidgets.QPushButton( "Seleccionar archivo CSV" , objectName="CSV-Upload-Button" )
        CardCSV.addWidget( self.CSVUploadButton )
        CSVUploadLayout.addWidget( self.CSVUploadButton )

        self.AnalizeButton = QtWidgets.QPushButton( "Analizar datos" , objectName="Analize-Button" )
        self.AnalizeButton.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.AnalizeButton.setEnabled( False )
        CSVUploadLayout.addWidget( self.AnalizeButton )


        CSVLayout.addWidget( CardCSV )


# ==================== Layout de Información  ====================

        InfoLayout = QtWidgets.QHBoxLayout()
        InfoLayout.setContentsMargins(0, 0, 0, 0)
        ContentLayout.addLayout( InfoLayout )

# ==================== Layout de métricas ====================

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName("ScrollArea")
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scrollArea.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        scrollArea.setContentsMargins(0, 0, 0, 0)
        # scrollArea.setHorizontalScrollBarPolicy(QtCore.ScrollBarAlwaysOff)
        # scrollArea.setVerticalScrollBarPolicy(QtCore.ScrollBarAlwaysOn)
        # scrollArea.setWidgetResizable(True)
        InfoLayout.addWidget( scrollArea )

        contentWidget = QtWidgets.QWidget()
        scrollArea.setWidget(contentWidget)

        MetricsLayout = QtWidgets.QVBoxLayout()
        MetricsLayout.setContentsMargins(25, 13, 25, 25)
        MetricsLayout.setSpacing(25)

        contentWidget.setLayout( MetricsLayout )
        # InfoLayout.addLayout( MetricsLayout )

        CardProductividad = CartaMetricas('Productividad', 'Mérito productivo')
        CardProductividad.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        MetricsLayout.addWidget( CardProductividad )

        CardComportamiento = CartaMetricas('Comportamiento', 'Riesgo de comportamiento')
        CardComportamiento.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        MetricsLayout.addWidget( CardComportamiento )

        CardSalud = CartaMetricas('Sanidad', 'Riesgo de sanidad')
        CardSalud.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        MetricsLayout.addWidget( CardSalud )

# ==================== Layout de resultados ====================

        ResultsLayout = QtWidgets.QVBoxLayout()
        ResultsLayout.setContentsMargins(25, 13, 25,25)
        InfoLayout.addLayout( ResultsLayout )


        Card5 = Carta()
        ResultsLayout.addWidget( Card5 )
