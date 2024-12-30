import os
import json
import shutil

folder_zdj = "sciezka_do_zdjec"
folder_dodane = "sciezka_docelowa"

sciezka_do_json = "adnotacje.json"

try:
    with open(sciezka_do_json, "r") as plik:
        dane = json.load(plik)
except FileNotFoundError:
    print(f"Plik {sciezka_do_json} nie został znaleziony.")
    exit()
except json.JSONDecodeError as e:
    print(f"Błąd w formacie pliku JSON: {e}")
    exit()

if not os.path.exists(folder_dodane):
    os.makedirs(folder_dodane)

for element in dane:
    nazwa_zdjecia = element.get("image")
    if nazwa_zdjecia:
        sciezka_zrodlowa = os.path.join(folder_zdj, nazwa_zdjecia)
        sciezka_docelowa = os.path.join(folder_dodane, nazwa_zdjecia)

        if os.path.exists(sciezka_zrodlowa):
            try:
                shutil.move(sciezka_zrodlowa, sciezka_docelowa)
                print(f"Przeniesiono: {nazwa_zdjecia}")
            except Exception as e:
                print(f"Błąd podczas przenoszenia {nazwa_zdjecia}: {e}")
        else:
            print(f"Plik {nazwa_zdjecia} nie istnieje w folderze {folder_zdj}.")
    else:
        print("Nie znaleziono nazwy zdjęcia w jednym z elementów JSON.")
