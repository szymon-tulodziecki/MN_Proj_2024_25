import json
import os
import shutil

with open('adnotacje.json', 'r') as f:
    dane = json.load(f)

nazwy_plikow_obrazow = [element['obraz'] for element in dane]

katalog_zrodlowy = 'zdj'
katalog_docelowy = 'dodane'

os.makedirs(katalog_docelowy, exist_ok=True)

for nazwa_pliku in nazwy_plikow_obrazow:
    sciezka_zrodlowa = os.path.join(katalog_zrodlowy, nazwa_pliku)
    sciezka_docelowa = os.path.join(katalog_docelowy, nazwa_pliku)
    if os.path.exists(sciezka_zrodlowa):
        shutil.move(sciezka_zrodlowa, sciezka_docelowa)
        print(f'Przeniesiono: {nazwa_pliku}')
    else:
        print(f'Plik nie znaleziony: {nazwa_pliku}')