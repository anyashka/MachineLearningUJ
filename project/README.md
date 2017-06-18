## Wstęp

Pierwsza część projektu znajduje się w folderze 'Part 1', druga odpowiednio w 'Part 2'. Kod z rozwiązaniem znajduje się w pliku 'train.py'.

## Część pierwsza

- Jako model były wybrane sieci konwolucyjne i zaimplemetowane za pomocą bibilioteki Keras.
Było wybrano taką bibliotekę biorąc pod uwagę to, że model jest bardzo łatwy w implementacji z pomocą Keras.
Użyłam sieci konwolucyjnych, bo one są jednym z najlepszych rozwiązań problemu klasyfikacji obrazków.
Najpierw w sieciach konwolucyjnych jest stosowany filtr to poszególnych fragmentów obrazku. Sieci używają tego samego filtru dla każdego pikselu w warstwie. Za pomocą zastosowania takiego filtru są uzyskiwane 'features', czyli wlaściwości obrazku. Warstwa 'poolingu' pomaga progresywnie zmniejszyć rozmiar reprezetacji, liczbę parametrów i szybkość obliczenia. 

- Opis hiperparametrów:
	⋅⋅* **batch_size** — ilość obrazków trenujących, które są wykorzystane jednocześnie pod czas jednej iteracji
	⋅⋅* **num_epochs** — ilość iteracji algorytmu trenującego po całym zbiorze
	⋅⋅* **kernel_size** — rozmiar jądra w odpowiedniej warstwie
	⋅⋅* **pool_size** — rozmiar podzbioru w warstwie poolingu
	⋅⋅* **сonv_depth** — ilość jąder w konwolucyjnych warstwach 
	⋅⋅* **drop_prob** (dropout probability) — ilość w procentach ile neuronów będzie usunętych w warstwie 'dropout' 
	⋅⋅* **hidden_size** — ilość neuronów w warstwie Dense

	można też dobrać:
	⋅⋅* **activation** - funckcja aktywacyjna (ReLU, Sigmoid i t.d.)
	⋅⋅* **optimizer** - algorytm optymizacji 
	⋅⋅ i t.d.

Dla wyboru hiperparametrów można skorzystać z: Grid Search, Bayesian optimization, Random Search i Gradient-based Optimization.
Ja skorzystałam z Grid Search. Kod można zobaczyć w pliku 'Part 1/GridSearchCV.py'. W celu szybkiego działania była zmniejszona ilość obrazków trenujących, jednak sama sieć konwolucyjna nie zmieniła się przy stosowaniu pełnych danych.
Podając różne zestawy hiperparametrów uzyskałam następne wyniki:
```
Best: 0.180000 using {'activation': 'relu', 'neurons': 512, 'optimizer': 'adam'}
0.130000 (0.027495) with: {'activation': 'relu', 'neurons': 8, 'optimizer': 'adam'}
0.170000 (0.048454) with: {'activation': 'relu', 'neurons': 8, 'optimizer': 'adagrad'}
0.140000 (0.039140) with: {'activation': 'relu', 'neurons': 8, 'optimizer': 'sgd'}
0.180000 (0.026656) with: {'activation': 'relu', 'neurons': 512, 'optimizer': 'adam'}
0.160000 (0.015420) with: {'activation': 'relu', 'neurons': 512, 'optimizer': 'adagrad'}
0.110000 (0.037333) with: {'activation': 'relu', 'neurons': 512, 'optimizer': 'sgd'}
0.100000 (0.050840) with: {'activation': 'sigmoid', 'neurons': 8, 'optimizer': 'adam'}
0.110000 (0.013477) with: {'activation': 'sigmoid', 'neurons': 8, 'optimizer': 'adagrad'}
0.130000 (0.035553) with: {'activation': 'sigmoid', 'neurons': 8, 'optimizer': 'sgd'}
0.100000 (0.027686) with: {'activation': 'sigmoid', 'neurons': 512, 'optimizer': 'adam'}
0.110000 (0.038739) with: {'activation': 'sigmoid', 'neurons': 512, 'optimizer': 'adagrad'}
0.140000 (0.013311) with: {'activation': 'sigmoid', 'neurons': 512, 'optimizer': 'sgd'}
```
Dlatego sieć konwolucyjna była nauczona z najlepszymi parametrami.

Najpiew Grid Search był wywołany dla rzeczywistego zbioru i z większą liczbą różnych hyperparametrów. Ale niestety z powodu ograniczonych możliwości sprzętu i ograniczonego czasu ten wynik jeszcze nie został uzyskany. 

## Część druga

Dla wykonania unsupervised pretrainingu było użyto autoencodera. W najprostszym wariancie autoencoder posiada warswę wejściową, pośrednią oraz wyjściową. Rozmiar warstwy wejściowej i wyjściowej jest identyczny. Stosując autoencoder ale ilości epoch = 15 zauważano, że nie wiele to zmienia:
```
Epoch 1/15
50000/50000 [==============================] - 424s - loss: 0.5940 - val_loss: 0.5757
Epoch 5/15
50000/50000 [==============================] - 421s - loss: 0.5654 - val_loss: 0.5657
Epoch 10/15
Epoch 15/15
50000/50000 [==============================] - 428s - loss: 0.5635 - val_loss: 0.5636
```
Dla tego wynik był generowany za pomocą autoencodera z ilością epoch = 5. 

Porównując wyniki modelu z pretrainingem oraz modelu bez pretrainingu, można zauważyć, że model z autoencoderem liczył się szybcziej. 

```
'Train accuraccy of model with pretraining:', 0.67700000000000005
'Train accuraccy of model without pretraining:', 0.66359999999999997
```
Jak widać, training accuraccy dla modelu z pretrainingem jest nieznacznie, ale lepsza. 
