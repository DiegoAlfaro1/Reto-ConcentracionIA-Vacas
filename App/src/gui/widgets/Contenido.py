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
        ContentLayout.setContentsMargins(25, 25, 25, 25)
        ContentLayout.setSpacing(25)
        self.setLayout(ContentLayout)
        self.setObjectName("Content-Background")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.MinimumExpanding)



# ==================== Layout para subir el CSV ====================

        CSVLayout = QtWidgets.QHBoxLayout()
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

        CSVUploadButton = QtWidgets.QPushButton( "Seleccionar archivo CSV" , objectName="CSV-Upload-Button" )
        CardCSV.addWidget( CSVUploadButton )
        CSVUploadLayout.addWidget( CSVUploadButton )

        AnalizeButton = QtWidgets.QPushButton( "Analizar datos" , objectName="Analize-Button" )
        AnalizeButton.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)

        CSVUploadLayout.addWidget( AnalizeButton )


        CSVLayout.addWidget( CardCSV )


# ==================== Layout de Información  ====================

        InfoLayout = QtWidgets.QHBoxLayout()
        ContentLayout.addLayout( InfoLayout )

# ==================== Layout de métricas ====================

        MetricsLayout = QtWidgets.QVBoxLayout()
        MetricsLayout.setSpacing(25)
        InfoLayout.addLayout( MetricsLayout )

        CardProductividad = CartaMetricas('Productividad', 'Mérito productivo')
        MetricsLayout.addWidget( CardProductividad )

        CardComportamiento = Carta()
        MetricsLayout.addWidget( CardComportamiento )

        CardSalud = Carta()
        MetricsLayout.addWidget( CardSalud )

# ==================== Layout de resultados ====================

        ResultsLayout = QtWidgets.QVBoxLayout()
        InfoLayout.addLayout( ResultsLayout )


        Card5 = Carta()
        ResultsLayout.addWidget( Card5 )
