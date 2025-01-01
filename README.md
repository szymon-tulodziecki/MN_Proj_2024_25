### Wybrany obiekt do klasyfikacji:
- **Samochody**  

### Link do repozytorium projektu:
- [Projekt na GitHub](https://github.com/szymon-tulodziecki/MN_Proj_2024_25)

---


## 2. Zdjęcia:

- **Źródło zdjęć**: [Roboflow Universe](https://universe.roboflow.com).
- **Format nazw**: `1.jpg` - `330.jpg` (treningowe), `t1.jpg` - `t10.jpg` (testowe).
- **Linki do zbiorów**:
  - [Treningowy](https://github.com/szymon-tulodziecki/MN_Proj_2024_25/blob/main/dodane.tar.gz)
  - [Testowy](https://github.com/szymon-tulodziecki/MN_Proj_2024_25/blob/main/test.tar.gz)

---

## 3. Etykietowanie obiektów:

- **Aplikacja etykietująca**: [Kod aplikacji](https://github.com/szymon-tulodziecki/MN_Proj_2024_25/blob/main/aplikacja_do_etykietowania.py).  
  - Graficzny interfejs użytkownika (GUI) stworzony w PyQt5.
  - Dane zapisywane w formacie JSON z etykietami prostokątnymi, zgodne z wymogami PyTorch.
  - Funkcjonalności: wybór folderu z obrazami, dodawanie etykiet, zapis, reset, obsługa błędów.

---

## 4. Eksport danych do Dataset PyTorch:

- **Kod eksportu**: [Plik trenowanie.py](https://github.com/szymon-tulodziecki/MN_Proj_2024_25/blob/main/trenowanie.py).  
- Użyte biblioteki: `torch`, `torchvision`.
- Format przetwarzanych danych: `[x_min, y_min, x_max, y_max]`.
- Konwersja na tensory dla zgodności z modelami PyTorch.

---

## 5. Trenowanie modelu Faster R-CNN:

- **Użyte modele**: ResNet-50, ResNet-101.
- **Kod**: [Plik trenowanie.py](https://github.com/szymon-tulodziecki/MN_Proj_2024_25/blob/main/trenowanie.py).
- **Platforma**: [Google Colab](https://colab.google/) z wykorzystaniem GPU.
- **Przebieg treningu**:
  - Zbiór treningowy: 80%, walidacyjny: 20%.
  - Modele przetrenowane przez 10 epok.
  - Wyniki zapisane do plików `.pth` dla późniejszego wykorzystania.

---

## 6. Testowanie modeli Faster R-CNN:

- **Kod testowy**: [Plik testowanie.py](https://github.com/szymon-tulodziecki/MN_Proj_2024_25/blob/main/testowanie.py).
- Funkcjonalności:
  - Testy na zbiorze testowym z progami pewności.
  - Usuwanie nakładających się ramek.
  - Wizualizacja wyników.

---

## 7. Wyniki i wnioski:

- **Model ResNet-50**:
  - Średni czas predykcji: 0.0832 sekundy.
  - Średnia wartość predykcji: 0.9977.
- **Model ResNet-101**:
  - Średni czas predykcji: 0.0946 sekundy.
  - Średnia wartość predykcji: 0.9821.
- **Wnioski**:
  - ResNet-50 lepiej radzi sobie z krótszym czasem i większą precyzją.
  - Większy zbiór danych mógłby poprawić wyniki.

---

## 8. Źródła:

1. [GeeksforGeeks: Image Datasets, Dataloaders, and Transforms in PyTorch](https://www.geeksforgeeks.org/image-datasets-dataloaders-and-transforms-in-pytorch/)
2. [PyTorch Vision Documentation](https://pytorch.org/vision/stable/index.html)
3. [PyTorch Pre-trained Models](https://pytorch.org/vision/stable/models.html#classification)
4. [Roboflow Universe](https://universe.roboflow.com/)
5. [Stanford Car Dataset Repository](https://github.com/sigopt/stanford-car-classification?tab=MIT-1-ov-file)
6. [Learn OpenCV: PyTorch for Beginners](https://learnopencv.com/pytorch-for-beginners-basics/)
7. [YouTube Tutorial: PyTorch Faster R-CNN](https://www.youtube.com/watch?v=oh7UO4IoAls)
8. [GitHub: LabelImg](https://github.com/HumanSignal/labelImg)
9. [Google Colab](https://colab.google/)

---

