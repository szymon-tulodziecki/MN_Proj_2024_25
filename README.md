Projekt jest realizowany w grupach maksymalnie 3 osobowych. Należy zapisać się w kursie do grupy. Każda grupa pracuje na zestawie minimum 300 zdjęć. 
Wykonaj następujące zadania:

    - Przygotuj folder z zapisanymi minimum 300 zdjęciami. Zdjęcia możesz pozyskać z udostępnionego zbioru, lub samodzielnie. 
    - W Twoim projekcie danymi wejściowymi będzie zbiór obrazów. W ramach projektu dla każdego obrazu powinieneś przygotować plik zawierający etykiety obiektów wybranej klasy (np. osoba, znak drogowy, adres budynku itd.)
    - W rezultacie dla zbioru 300 zdjęć należy stworzyć polik lub pliki z etykietami, zawierający zbiór współrzędnych (etykietę) oraz nazwę obiektów danej klasy znajdujących się na konkretnym zdjęciu.
    - W zbiorze zdjęć samodzielnie wyodrębnij klasę obiektów, które będą oznaczane - czyli etykietowane.
    - Do etykietowania możesz wykorzystać dowolne narzędzia programistyczne. Projekt będzie wyżej oceniany jeśli stworzysz i zaprezentujesz własny sposób etykietowania. Finalnie sposób etykietowania powinien być użyteczny przy pracy z biblioteką Pytorch.
    - Ważne jest, aby przygotowany zbiór z etykietami był kompatybilny z Pytorch. Wykonaj eksport Twojego zbioru do Dataset w Pytorch, np tak jak opisano tutaj https://www.geeksforgeeks.org/image-datasets-dataloaders-and-transforms-in-pytorch/
    - Wykonaj przykład klasyfikacji (wyodrębnienie obiektów graficznych) w Pytorch TorchVision, korzystając z opracowanego zbioru. W dokumentacji znajdziesz informacje dot. modeli, przykłady użycia modeli oraz szczegóły nt. przygotowania zbioru danych: https://pytorch.org/vision/stable/index.html
    - Na minimalnym poziomie wykonania projektu przy klasyfikacji możesz użyć przetrenowanych modeli: https://pytorch.org/vision/stable/models.html#classification
    - Porównaj ze sobą rezultaty uzyskane przy użyciu 5 modeli.
    - Na maksymalnym poziomie wykonania projektu należy samodzielnie wykonać trenning wybranego modelu (jednego lub więcej) i na nim pokazać procedurę klasyfikacji wybranych obiektów w obrazach.
    - W raporcie przedstaw analizę jakości uzyskanego rezultatu. Możesz również przedstawić narzędzia (programy, biblioteki), które zostały  wykorzystane.



1. aplikacja_do_etykietowania.py jest to program do etykietowania zdjęć, po uruchomieniu oczekiwane jest wybranie folderu ze zdjęciami. Następnie inicjowana jest graficzna aplikacja, która pozwala na nakładanie ramek/etykiet
w obrębie zdjęcia za pomocą suwaków. Współrzędne obiektu zapisywane są w pliku adnotacje.jsos (x, y, dlugosc, szerokosc), gdzie x, y są współrzędnymi lewego górnego rogu etykiety
2. Przygotowany plik adnotacje.json pozwala na wykonanie dataset'u i trenowanie na jego podstawie modelu (FastRCNN). Tak przetrenowany model jest zapisywany, w celu uniknięcia konieczności wielokrotnego powtarzania tej czynności (plik: trenowanie.py)
3. Przetrenowany model można użyć do sprawdzenia jego działania. Plik testowanie.py uruchamia przetrenowany model na zdjeciach testowych i zwraca je z oznczeniami
