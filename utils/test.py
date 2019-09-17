import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QWidget, QLCDNumber, QSlider,
    QVBoxLayout, QApplication)
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQuick import QQuickView
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

class Example(QWidget):

    def __init__(self):
        super().__init__()
        self.main = None
        self.opengl_w = None
        self.upd_t = None
        self.qml = None
        self.initUI()


    def getQML(self):
        qqml = QQuickView()
        qqml.setSource(QUrl('visual.qml'))
        self.qml = QWidget.createWindowContainer(qqml, self)
        return self.qml

    def getOPENGLWIDGET(self):
        self.opengl_w = gl.GLViewWidget()
        self.opengl_w.show()
        self.opengl_w.setWindowTitle('pyqtgraph example: GLSurfacePlot')
        self.opengl_w.setCameraPosition(distance=50)
        self.opengl_w.setBackgroundColor((255,255,255,100))

        ## Add a grid to the view
        g = gl.GLGridItem()

        g.setColor((0,0,0,255))
        g.scale(2, 2, 1) # draw grid after surfaces since they may be translucent
        self.opengl_w.addItem(g)

        return self.opengl_w

    def initUI(self):

        vbox = QVBoxLayout()

        a = self.getOPENGLWIDGET()
        b = self.getQML()
        vbox.addWidget(a)
        vbox.addWidget(b)

        self.setLayout(vbox)
        self.setGeometry(500, 300, 600, 600)
        self.setWindowTitle('Signal & slot')
        self.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())