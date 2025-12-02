from PyQt6 import QtWidgets, QtCore, QtGui

class Carta( QtWidgets.QWidget ):
    def __init__( self, parent = None):
        super().__init__( parent )

        self.setupUI()

    def setupUI( self ):

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

    def addWidget( self, widget : QtWidgets.QWidget ):
        self.layout().addWidget( widget )

    def addLayout( self, layout : QtWidgets.QLayout ):
        self.layout().addLayout( layout )
