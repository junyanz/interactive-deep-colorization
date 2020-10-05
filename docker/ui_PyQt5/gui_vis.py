from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
from data import lab_gamut
import pdb

try:
    QString = unicode
except NameError:
    # Python 3
    QString = str

try:
    QString = unicode
except NameError:
    # Python 3
    QString = str


class GUI_VIS(QWidget):

    update_color = pyqtSignal(QString)


    def __init__(self, win_size=256, scale=2.0):
        QWidget.__init__(self)
        self.result = None
        self.win_width = win_size
        self.win_height = win_size
        self.scale = scale
        self.setFixedSize(self.win_width, self.win_height)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), QColor(49, 54, 49))
        if self.result is not None:
            h, w, c = self.result.shape
            qImg = QImage(self.result.tostring(), w, h, QImage.Format_RGB888)
            dw = int((self.win_width - w) // 2)
            dh = int((self.win_height - h) // 2)
            painter.drawImage(dw, dh, qImg)

        painter.end()

    def update_result(self, result):
        self.result = result
        self.update()

    def sizeHint(self):
        return QSize(self.win_width, self.win_height)

    def reset(self):
        self.update()
        self.result = None

    def is_valid_point(self, pos):
        if pos is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            return x >= 0 and y >= 0 and x < self.win_width and y < self.win_height

    def scale_point(self, pnt):
        x = int(pnt.x() / self.scale)
        y = int(pnt.y() / self.scale)
        return x, y

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L = self.im_l[y, x]
        snap_color = lab_gamut.snap_ab(mean_L, color_array)
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def mousePressEvent(self, event):
        pos = event.pos()
        x, y = self.scale_point(pos)
        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            if self.result is not None:
                color = self.result[y, x, :]  #
                c = QColor(color[0], color[1], color[2])
                # color = self.calibrate_color(c, self.pos)
                self.update_color.emit(QString('background-color: %s' % c.name()))
                print('color', color)

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass
