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

        self.CardProductividad = CartaMetricas('Productividad', 'Mérito productivo')
        self.CardProductividad.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        MetricsLayout.addWidget( self.CardProductividad )

        self.CardComportamiento = CartaMetricas('Comportamiento', 'Riesgo de comportamiento')
        self.CardComportamiento.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        MetricsLayout.addWidget( self.CardComportamiento )

        self.CardSalud = CartaMetricas('Sanidad', 'Riesgo de sanidad')
        self.CardSalud.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        MetricsLayout.addWidget( self.CardSalud )

# ==================== Layout de resultados ====================

        InterpretationLayout = QtWidgets.QVBoxLayout()
        InterpretationLayout.setContentsMargins(25, 13, 25,25)
        InterpretationLayout.setSpacing(25)
        InfoLayout.addLayout( InterpretationLayout )

        ResultsLayout = QtWidgets.QHBoxLayout()
        InterpretationLayout.addLayout( ResultsLayout )

        CardIMR = Carta()
        CardIMR.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        ResultsLayout.addWidget( CardIMR )

        IMRLayout = QtWidgets.QHBoxLayout()
        IMRLayout.setContentsMargins(0, 0, 0, 0)
        IMRLayout.setSpacing(0)
        CardIMR.addLayout( IMRLayout )

        IMRLabel = QtWidgets.QLabel( 'IMR', objectName="Card-Title")
        self.IMR = QtWidgets.QLabel( '-', objectName="Card-Text")
        
        IMRLayout.addWidget(  IMRLabel )
        IMRLayout.addWidget( self.IMR, alignment=aflag.AlignRight )

        self.DecitionCard = Carta()
        self.DecitionCard.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        self.DecitionCard.setObjectName("Card-Decition")
        DecitionLayout = QtWidgets.QHBoxLayout()
        DecitionLabel = QtWidgets.QLabel( 'Decisión', objectName="Card-Title")
        self.Decition = QtWidgets.QLabel( '-', objectName="Card-Text")

        DecitionLayout.addWidget(  DecitionLabel )
        DecitionLayout.addWidget( self.Decition, alignment=aflag.AlignRight )
        self.DecitionCard.addLayout( DecitionLayout )

        ResultsLayout.addWidget( self.DecitionCard )



        Card5 = Carta()
        InterpretationLayout.addWidget( Card5 )
