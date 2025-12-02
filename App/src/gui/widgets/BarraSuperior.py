from PyQt6 import QtWidgets, QtCore, QtGui

class BarraSuperior( QtWidgets.QWidget ):
    def __init__( self, parent = None):
        super().__init__( parent )

        self.setupUI()

    def setupUI( self ):

        aflag = QtCore.Qt.AlignmentFlag
        
        # Contenedores para la barra de arriba con los textos y cuadro con información
        TopBarLayout = QtWidgets.QHBoxLayout()
        TopBarLayout.setContentsMargins(50, 50, 50, 50)
        TopBarLayout.setSpacing(10)
        self.setLayout(TopBarLayout)
        self.setObjectName("Top-Bar-Background")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)

        TextLayout = QtWidgets.QVBoxLayout()
        # CSVLayout = QtWidgets.QVBoxLayout()

        TextLayout.setContentsMargins(0, 0, 0, 0)
        TextLayout.setSpacing(10)

        PanelOperativoTitle = QtWidgets.QLabel( 
            "Panel Operativo",
            objectName="Title-Panel-Operativo")
        PanelOperativoTitle.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        PanelOperativoTitle.setWordWrap(True) 
        TextLayout.addWidget( PanelOperativoTitle )

        PanelOperativoSubtitle = QtWidgets.QLabel( 
            "Top de vacas según productividad, salud y genética", 
            objectName="Subtitle-Panel-Operativo")
        PanelOperativoSubtitle.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        PanelOperativoSubtitle.setWordWrap(True) 
        TextLayout.addWidget( PanelOperativoSubtitle )

        PanelOperativoDescription = QtWidgets.QLabel( 
            "Explora el desempeño histórico de cada vaca y detecta aquellas con mayor potencial para los programas de reproducción y ordeño.", 
            objectName="Description-Panel-Operativo")
        PanelOperativoDescription.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        PanelOperativoDescription.setWordWrap(True) 
        TextLayout.addWidget( PanelOperativoDescription )

        TopBarLayout.addLayout( TextLayout )

