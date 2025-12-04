from PyQt6 import QtWidgets, QtCore, QtGui

class Interpretation( QtWidgets.QWidget ):
    def __init__( self, parent = None):
        super().__init__( parent )

        self.setupUI()

    def setupUI( self ):

        aflag = QtCore.Qt.AlignmentFlag
        
        # Contenedores para la barra de arriba con los textos y cuadro con información
        InterpretationLayout = QtWidgets.QVBoxLayout()
        InterpretationLayout.setContentsMargins(0, 0, 0, 0)
        InterpretationLayout.setSpacing(0)
        # InterpretationLayout.setMinimumHeight(0)
        # InterpretationLayout.setMaximumHeight(50)
        self.setLayout(InterpretationLayout)


        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName("ScrollArea2")
        scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scrollArea.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        scrollArea.setContentsMargins(0, 0, 0, 0)
        # scrollArea.setFixedHeight(100)

        InterpretationLayout.addWidget( scrollArea )

        contentWidget = QtWidgets.QWidget()
        scrollArea.setWidget(contentWidget)

        TextLayout = QtWidgets.QVBoxLayout()
        TextLayout.setContentsMargins(0, 0, 0, 0)
        TextLayout.setSpacing(10)

        InterpretationTitle = QtWidgets.QLabel( 'Interpretación de métricas', objectName="Card-Title")
        InterpretationTitle.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        TextLayout.addWidget( InterpretationTitle )
        # self.Decition = QtWidgets.QLabel( '-', objectName="Card-Text")

        hline = QtWidgets.QFrame()
        hline.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline.setObjectName("Line")
        TextLayout.addWidget( hline )


        text1 = """Nuestro algoritmo compara a cada vaca individual contra el promedio histórico de todo tu hato para darte una recomendación objetiva."""

        text2 = """1. Entendiendo el "Valor Z" (La Comparativa) Este valor no mide kilos o litros, sino qué tan diferente es esta vaca respecto al promedio de sus compañeras.

    En Productividad: Aquí buscamos valores Positivos (+). Un valor positivo (ej. +1.17) indica que la vaca está por encima del promedio. Mientras más alto sea el número, más productiva es comparada con el resto.

    En Riesgos (Salud y Comportamiento): Aquí buscamos valores Negativos (-). Un valor negativo (ej. -3.31) es excelente, porque significa que la vaca tiene menos incidencias o problemas que la vaca promedio.

        Positivo en riesgo = Vaca problemática.

        Negativo en riesgo = Vaca sana/dócil. """

        text3 = """2. IMR (Índice de Mérito para Retención) Es la calificación final que equilibra la balanza. Toma los puntos buenos de la producción y le resta los puntos malos de los riesgos.

    El objetivo: Queremos un IMR alto (cercano o superior a 1). Esto nos dice que la producción de la vaca justifica con creces cualquier costo asociado a su mantenimiento."""

        text4 = """3. Decisión Sugerida Para facilitar tu trabajo, el sistema traduce el IMR en acciones claras:

    Retener / Reproducir: La vaca es del Top 25%. Prioridad genética.

    Supervisar: Vaca promedio. Mantenerla, pero vigilar su evolución.

    Descartar: Vaca del 40% inferior. Su desempeño es bajo comparado con el resto y es candidata a salir."""


        InterpretationContent1 = QtWidgets.QLabel( text1, objectName="Card-Text")
        InterpretationContent1.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        InterpretationContent1.setWordWrap(True)
        TextLayout.addWidget( InterpretationContent1 )

        hline1 = QtWidgets.QFrame()
        hline1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline1.setObjectName("Line")
        TextLayout.addWidget( hline1 )

        InterpretationContent2 = QtWidgets.QLabel( text2, objectName="Card-Text")
        InterpretationContent2.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        InterpretationContent2.setWordWrap(True)
        TextLayout.addWidget( InterpretationContent2 )

        hline2 = QtWidgets.QFrame()
        hline2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline2.setObjectName("Line")
        TextLayout.addWidget( hline2 )

        InterpretationContent3 = QtWidgets.QLabel( text3, objectName="Card-Text")
        InterpretationContent3.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        InterpretationContent3.setWordWrap(True)
        TextLayout.addWidget( InterpretationContent3 )

        hline3 = QtWidgets.QFrame()
        hline3.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hline3.setObjectName("Line")
        TextLayout.addWidget( hline3 )

        InterpretationContent4 = QtWidgets.QLabel( text1, objectName="Card-Text")
        InterpretationContent4.setAlignment( aflag.AlignLeft | aflag.AlignTop )
        InterpretationContent4.setWordWrap(True)
        TextLayout.addWidget( InterpretationContent4 )

        contentWidget.setLayout( TextLayout )