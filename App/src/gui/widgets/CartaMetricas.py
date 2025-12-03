from PyQt6 import QtWidgets, QtCore, QtGui

from src.gui.widgets.Carta import Carta


class CartaMetricas( QtWidgets.QWidget ):
    def __init__( self, titulo, metrica, parent = None):
        super().__init__( parent )
        self.titulo = titulo
        self.metrica = metrica
        self.setupUI()

    def setupUI( self ):
        aflag = QtCore.Qt.AlignmentFlag

        aflag = QtCore.Qt.AlignmentFlag
        
        # Contenedores para la barra de arriba con los textos y cuadro con información
        CardLayout = QtWidgets.QVBoxLayout()
        CardLayout.setContentsMargins(25, 25, 25, 25)
        CardLayout.setSpacing(5)
        self.setLayout(CardLayout)
        self.setObjectName("Card")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)       # que tan difuminada es la sombra
        shadow.setXOffset(0)           # desplazamiento horizontal
        shadow.setYOffset(0)           # desplazamiento vertical
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))  # color con alpha para traslucidez

        self.setGraphicsEffect(shadow)


        # título

        CardTitle = QtWidgets.QLabel( 
            self.titulo,
            objectName="Card-Title")
        CardTitle.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        CardTitle.setWordWrap(True) 
        CardLayout.addWidget(  CardTitle )

        # Línea divisora

        hline = QtWidgets.QFrame()
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setObjectName("Line")

        CardLayout.addWidget( hline )

        # Métrica

        MetricLayout = QtWidgets.QHBoxLayout()

        MetricLabel = QtWidgets.QLabel( self.metrica, objectName="Card-Text")
        Metric = QtWidgets.QLabel( '5.32338', objectName="Card-Text")
        
        MetricLayout.addWidget(  MetricLabel )
        MetricLayout.addWidget( Metric, alignment=aflag.AlignRight )
        CardLayout.addLayout(  MetricLayout )

        # Línea divisora

        hline1 = QtWidgets.QFrame()
        hline1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline1.setObjectName("Line")
        CardLayout.addWidget( hline1 )

        # Valor Z

        ZValueLayout = QtWidgets.QHBoxLayout()

        ZValueLabel = QtWidgets.QLabel( 'Valor Z', objectName="Card-Text")
        ZValue = QtWidgets.QLabel( '5.32338', objectName="Card-Text")
        
        ZValueLayout.addWidget(  ZValueLabel )
        ZValueLayout.addWidget( ZValue, alignment=aflag.AlignRight )
        CardLayout.addLayout(  ZValueLayout )

