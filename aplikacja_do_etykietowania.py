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
        """" Metoda klasy usprawniająca pracę z etykietą (rozmiar, przesuwanie itp.) """
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
        """ Metoda klasy pozwalająca na przesuwanie etykiety """
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
        """ Metoda klasy pozwalająca na zmianę rozmiaru etykiety """
        self.zmiana_rozmiaru = False
        super().mouseReleaseEvent(event)
        if self.scene() and hasattr(self.scene(), 'aktualizuj_etykieta'):
            self.scene().aktualizuj_etykieta(self)

class AplikacjaDoOznaczaniaObrazow(QMainWindow):
    """ Klasa główna aplikacji """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikacja do Oznaczania Obrazów")
        self.setGeometry(100, 100, 1200, 800)

        self.folder_z_obrazami  QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami")
        if not self.folder_z_obrazami:
            sys.exit("Nie wybrano folderu z obrazami")

        self.lista_obrazow = [f for f in os.listdir(self.folder_obrazow) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.liczba_obrazow = len(self.lista_obrazow)
        self.indeks_obecnego_obrazu = 0
        self.etykiety = []

        self.plik_adnotacji = os.path.join(os.getcwd(), "adnotacje.json")
        self.klasy_pojazdow = ['samochod', 'brak']
        self.kolory = ['#00FF99', '#FF6600', '#3366FF', '#FF0066', '#33FFCC', '#9900FF', '#CCCC00', '#FFCCFF']
        self.indeks_obecnego_koloru = 0
        self.wspolczynnik_zoom = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 2.0

        self.inicjalizujUI()

     def inicjalizujUI(self):
        glowny_uklad = QVBoxLayout()

        gorny_uklad = QHBoxLayout()
        naglowek = QLabel("Menu Opcji", self)
        naglowek.setStyleSheet("font-size: 18px; font-weight: bold;")
        gorny_uklad.addWidget(naglowek)

        self.wybor_klasy = QComboBox(self)
        self.wybor_klasy.addItems(self.klasy_pojazdow)
        self.wybor_klasy.setStyleSheet("padding: 5px;")
        gorny_uklad.addWidget(self.wybor_klasy)

        self.przycisk_dodaj_etykieta = QPushButton("Dodaj Etykieta", self)
        self.przycisk_dodaj_etykieta.setStyleSheet("background-color: #2c3e50; color: white; padding: 10px;")
        self.przycisk_dodaj_etykieta.clicked.connect(self.dodaj_etykieta)
        gorny_uklad.addWidget(self.przycisk_dodaj_etykieta)

        self.przycisk_resetuj = QPushButton("Resetuj Wszystkie Etykiety", self)
        self.przycisk_resetuj.setStyleSheet("background-color: #7C2A3E; color: white; padding: 10px;")
        self.przycisk_resetuj.clicked.connect(self.resetuj_etykiety)
        gorny_uklad.addWidget(self.przycisk_resetuj)

        self.przycisk_zapisz = QPushButton("Zapisz i Następny", self)
        self.przycisk_zapisz.setStyleSheet("background-color: #5A6B7C; color: white; padding: 10px;")
        self.przycisk_zapisz.clicked.connect(self.zapisz_adnotacje)
        gorny_uklad.addWidget(self.przycisk_zapisz)

        glowny_uklad.addLayout(gorny_uklad)

        self.etykieta_licznika_obrazow = QLabel(f"Obraz: {self.indeks_obecnego_obrazu + 1} / {self.liczba_obrazow}", self)
        glowny_uklad.addWidget(self.etykieta_licznika_obrazow)

        self.etykieta_obrazu = QLabel(self)
        self.scena = QGraphicsScene(self)
        self.scena.aktualizuj_etykieta = self.aktualizuj_etykieta
        self.widok = QGraphicsView(self.scena)
        self.widok.setAlignment(Qt.AlignCenter)
        glowny_uklad.addWidget(self.widok)

        self.przycisk_zakoncz = QPushButton("Zakończ", self)
        self.przycisk_zakoncz.setStyleSheet("background-color: #4B3F72; color: white; padding: 10px;")
        self.przycisk_zakoncz.clicked.connect(self.close)
        glowny_uklad.addWidget(self.przycisk_zakoncz)

        glowny_widget = QWidget()
        glowny_widget.setLayout(glowny_uklad)
        self.setCentralWidget(glowny_widget)

        self.setStatusBar(QStatusBar(self))

        self.zaladuj_obraz()

    def zaladuj_obraz(self):
        if self.indeks_obecnego_obrazu < len(self.lista_obrazow):
            sciezka_obrazu = os.path.join(self.folder_obrazow, self.lista_obrazow[self.indeks_obecnego_obrazu])
            self.obraz = QImage(sciezka_obrazu)
            self.current_image = self.obraz
            pixmapa = QPixmap.fromImage(self.obraz)

            skalowana_pixmapa = pixmapa.scaled(self.widok.width(), self.widok.height(), Qt.KeepAspectRatio)

            self.scena.clear()
            self.scena.addPixmap(skalowana_pixmapa)
            self.widok.setScene(self.scena)

            self.wysrodkuj_obraz(skalowana_pixmapa)

            self.aktualizuj_licznik_obrazow()
        else:
            print("Wszystkie obrazy zostały przetworzone.")
            self.close()

    def wysrodkuj_obraz(self, pixmapa):
        prostokat_sceny = self.widok.sceneRect()
        szerokosc_obrazu, wysokosc_obrazu = pixmapa.width(), pixmapa.height()

        przesuniecie_x = (prostokat_sceny.width() - szerokosc_obrazu) / 2
        przesuniecie_y = (prostokat_sceny.height() - wysokosc_obrazu) / 2

        self.widok.setSceneRect(przesuniecie_x, przesuniecie_y, szerokosc_obrazu, wysokosc_obrazu)

    def aktualizuj_licznik_obrazow(self):

        self.etykieta_licznika_obrazow.setText(f"Obraz: {self.indeks_obecnego_obrazu + 1} / {self.liczba_obrazow}")

    def dodaj_etykieta(self):
        klasa = self.wybor_klasy.currentText()
        if klasa == 'brak':
            self.etykiety.append({
                'klasa': klasa,
                'element': None
            })
        else:
            etykieta = EtykietaDoZmianyRozmiaru(50, 50, 100, 100, self.kolory[self.indeks_obecnego_koloru])
            self.scena.addItem(etykieta)
            self.etykiety.append({
                'klasa': klasa,
                'element': etykieta
            })
            self.indeks_obecnego_koloru = (self.indeks_obecnego_koloru + 1) % len(self.kolory)

    def resetuj_etykiety(self):
        self.etykiety.clear()
        self.scena.clear()
        self.zaladuj_obraz()

    def aktualizuj_etykieta(self, element):
        for etykieta in self.etykiety:
            if etykieta['element'] == element:
                etykieta['prostokat'] = element.rect().getRect()

    def zapisz_adnotacje(self):
        if any(etykieta['klasa'] == 'samochod' for etykieta in self.etykiety) and not any(
                etykieta['element'] for etykieta in self.etykiety if etykieta['klasa'] == 'samochod'):
            self.statusBar().showMessage("Proszę dodać przynajmniej jedną etykietę dla 'samochod'!", 3000)
            return

        adnotacje = []
        for etykieta in self.etykiety:
            klasa = etykieta['klasa']
            if klasa == 'brak':
                prostokat = []
            else:
                rect = etykieta['element'].rect()
                x, y, szerokosc, wysokosc = rect.x(), rect.y(), rect.width(), rect.height()

                x = max(0, x)
                y = max(0, y)
                szerokosc = min(self.current_image.width() - x, szerokosc)
                wysokosc = min(self.current_image.height() - y, wysokosc)
                prostokat = [x, y, szerokosc, wysokosc]

            adnotacje.append({
                'klasa': klasa,
                'prostokat': prostokat
            })

        dane_obrazu = {
            'obraz': self.lista_obrazow[self.indeks_obecnego_obrazu],
            'adnotacje': adnotacje
        }

        if os.path.exists(self.plik_adnotacji):
            with open(self.plik_adnotacji, 'r') as f:
                istniejące_dane = json.load(f)
        else:
            istniejące_dane = []
        istniejące_dane.append(dane_obrazu)

        with open(self.plik_adnotacji, 'w') as f:
            json.dump(istniejące_dane, f, indent=4)

        self.indeks_obecnego_obrazu += 1
        self.etykiety.clear()
        self.zaladuj_obraz()

    def wheelEvent(self, event):

        delta = event.angleDelta().y()
        if delta > 0:
            if self.wspolczynnik_zoom < self.max_zoom:
                self.wspolczynnik_zoom += 0.1
        elif delta < 0:
            if self.wspolczynnik_zoom > self.min_zoom:
                self.wspolczynnik_zoom -= 0.1


        self.widok.setTransform(QTransform().scale(self.wspolczynnik_zoom, self.wspolczynnik_zoom))
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AplikacjaOznaczaniaObrazow()
    ex.show()
    sys.exit(app.exec_())