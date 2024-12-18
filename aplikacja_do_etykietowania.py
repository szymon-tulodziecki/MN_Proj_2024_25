import os 
import sys 
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, \
    QGraphicsScene, QGraphicsView, QComboBox, QWidget, QGraphicsRectItem, QGraphicsItem, QStatusBar
from PyQt5.QtGui import QPixmap, QImage, QColor, QPen, QTransform
from PyQt5.QtCore import Qt, QRectF


class Etykieta(QGraphicsRectItem):
    """ Klasa pozwalająca na utworzenie etykiety na obrazie """
    def __init__(self, x, y, szerokosc, wysokosc, kolol, parent = None):
        super().__init__(x, y, szerokosc, wysokosc, parent)
        self.setFlags(QGraphicsItem.ItemIsMSelectable | QGraphicsItem.ItemIsMovable | 
                      QGraphicsItem.ItemSendsGeometryChanges | QGraphicsItem.ItemIsFocusable)
        self.setPen(QPen(QColor(kolor), 2))
        self.setAcceptHoverEvents(True)
        self.zmiana_rozmiaru = False
        self.kierunek_zmiany_rozmiaru = None

    def hoverMoveEvent(self, event):
        super().hoverMoveEvent(event)
        prostokat = self.rect()
        margin = 5
        pos = event.pos()
        self.kierunek_zmiany_rozmiaru = None

        if abs(pos.x() - prostokat.left()) < margin:
            if abs(pos.y() - prostokat.top()) < margin:
                self.setCursor(Qt.SizeFDiagCursor)
                self.kierunek_zmiany_rozmiaru = 'lewy_gorny'
            elif abs(pos.y() - prostokat.bottom()) < margin:
                self.setCursor(Qt.SizeBDiagCursor)
                self.kierunek_zmiany_rozmiaru = 'lewy_dolny'
            else:
                self.setCursor(Qt.SizeHorCursor)
                self.kierunek_zmiany_rozmiaru = 'lewa'
        elif abs(pos.x() - prostokat.right()) < margin:
            if abs(pos.y() - prostokat.top()) < margin:
                self.setCursor(Qt.SizeBDiagCursor)
                self.kierunek_zmiany_rozmiaru = 'prawy_gorny'
            elif abs(pos.y() - prostokat.bottom()) < margin:
                self.setCursor(Qt.SizeFDiagCursor)
                self.kierunek_zmiany_rozmiaru = 'prawy_dolny'
            else:
                self.setCursor(Qt.SizeHorCursor)
                self.kierunek_zmiany_rozmiaru = 'prawa'
        elif abs(pos.y() - prostokat.top()) < margin:
            self.setCursor(Qt.SizeVerCursor)
            self.kierunek_zmiany_rozmiaru = 'gorna'
        elif abs(pos.y() - prostokat.bottom()) < margin:
            self.setCursor(Qt.SizeVerCursor)
            self.kierunek_zmiany_rozmiaru = 'dolna'
        else:
            self.setCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if self.zmiana_rozmiaru:
            self.zmiana_rozmiaru = False
            return
        if self.zmiana_rozmiaru:
            rect = self.rect()
            pos = event.pos()
            if self.kierunek_zmiany_rozmiaru == 'lewy_gorny':
                new_rect = QRectF(pos.x(), pos.y(), rect.right() - pos.x(), rect.bottom() - pos.y())
            elif self.kierunek_zmiany_rozmiaru == 'prawy_gorny':
                new_rect = QRectF(rect.left(), pos.y(), pos.x() - rect.left(), rect.bottom() - pos.y())
            elif self.kierunek_zmiany_rozmiaru == 'lewy_dolny':
                new_rect = QRectF(pos.x(), rect.top(), rect.right() - pos.x(), pos.y() - rect.top())
            elif self.kierunek_zmiany_rozmiaru == 'prawy_dolny':
                new_rect = QRectF(rect.left(), rect.top(), pos.x() - rect.left(), pos.y() - rect.top())
            elif self.kierunek_zmiany_rozmiaru == 'lewa':
                new_rect = QRectF(pos.x(), rect.top(), rect.right() - pos.x(), rect.height())
            elif self.kierunek_zmiany_rozmiaru == 'prawa':
                new_rect = QRectF(rect.left(), rect.top(), pos.x() - rect.left(), rect.height())
            elif self.kierunek_zmiany_rozmiaru == 'gorna':
                new_rect = QRectF(rect.left(), pos.y(), rect.width(), rect.bottom() - pos.y())
            elif self.kierunek_zmiany_rozmiaru == 'dolna':
                new_rect = QRectF(rect.left(), rect.top(), rect.width(), pos.y() - rect.top())
            self.setRect(new_rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.zmiana_rozmiaru = False
        super().mouseReleaseEvent(event)
        if self.scene() and hasattr(self.scene(), 'aktualizuj_etykieta'):
            self.scene().aktualizuj_etykieta(self)
