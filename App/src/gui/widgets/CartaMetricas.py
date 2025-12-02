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
        CardLayout.setSpacing(10)
        self.setLayout(CardLayout)
        self.setObjectName("Card")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)       # que tan difuminada es la sombra
        shadow.setXOffset(0)           # desplazamiento horizontal
        shadow.setYOffset(0)           # desplazamiento vertical
        shadow.setColor(QtGui.QColor(0, 0, 0, 40))  # color con alpha para traslucidez

        self.setGraphicsEffect(shadow)




        CardTitle = QtWidgets.QLabel( 
            self.titulo,
            objectName="Card-Title")
        CardTitle.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        CardTitle.setWordWrap(True) 
        CardLayout.addWidget(  CardTitle )

        hline = QtWidgets.QFrame()
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        
        hline.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        hline.setObjectName("Line")

        CardLayout.addWidget( hline )
        CardLayout.addWidget( hline )

        MetricLabel = QtWidgets.QLabel( 
            self.metrica,
            objectName="Card-Title")