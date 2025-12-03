from PyQt6 import QtWidgets, QtCore, QtGui

class Interpretation( QtWidgets.QWidget ):
    def __init__( self, parent = None):
        super().__init__( parent )

        self.setupUI()

    def setupUI( self ):

        aflag = QtCore.Qt.AlignmentFlag
        
        # Contenedores para la barra de arriba con los textos y cuadro con información
        InterpretationLayout = QtWidgets.QVBoxLayout()
        self.setLayout(InterpretationLayout)


        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName("ScrollArea")
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scrollArea.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        scrollArea.setContentsMargins(0, 0, 0, 0)

        InterpretationLayout.addWidget( scrollArea )