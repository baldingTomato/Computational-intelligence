import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()


""" a)
reshape -> Dodanie dodatkowego wymiaru (1) wskazuje, że obrazy są w odcieniach szarości (grayscale). Modele konwolucyjne oczekują 
danych w formacie (szerokość, wysokość, liczba kanałów), gdzie liczba kanałów wynosi 1 dla obrazów monochromatycznych.
Dodoatkowo mamy normalizację wartości pikseli do zakresu [0, 1]

argmax -> Wybiera indeks największej wartości w każdym wektorze one-hot.
"""

""" b)
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)) -> Warstwa ta używa 32 filtrów o wymiarach (3x3) do ekstrakcji cech z obrazu.
Każdy filtr przesuwa się (aplikuje operację konwolucji) po całym obrazie, tworząc mapę cech.
Funkcja aktywacji ReLU (relu) jest stosowana w celu wprowadzenia nieliniowości.

MaxPooling2D((2, 2)) ->  redukuje rozmiar przestrzenny każdej mapy cech.
Dla każdej 2x2 sekcji wybierany jest maksymalny piksel.

Flatten() -> Spłaszcza dane do jednowymiarowego wektora.

Dense(64, activation='relu') -> Warstwa w pełni połączona (fully connected) z 64 neuronami.
Każdy neuron ma swoje wagi i bias, a funkcja aktywacji ReLU (relu) jest stosowana.

Dense(10, activation='softmax') -> Warstwa w pełni połączona z 10 neuronami, odpowiadającymi każdej z 10 klas cyfr (0-9).
Funkcja aktywacji Softmax przekształca wyniki na prawdopodobieństwa, których suma wynosi 1.

Warstwy konwolucyjne (Conv2D): Ekstrahują lokalne wzorce i cechy, takie jak krawędzie czy kąty, z obrazów.
Pooling (MaxPooling2D): Redukuje rozmiar danych, eliminując mniej istotne szczegóły, co pomaga zapobiegać przeuczeniu.
Flatten: Konwertuje dane przestrzenne na wektor jednowymiarowy, aby mogły być przetwarzane przez warstwy gęste.
Warstwy gęste (Dense): Uczą się skomplikowanych zależności między cechami a klasami, aby model mógł dokonywać trafnych predykcji.
Softmax: Produkuje prawdopodobieństwa przynależności do każdej z klas, umożliwiając klasyfikację.
"""


""" c)
Cyfra "8" jest mylona z cyfrą "9" w 10 przypadkach.
Cyfra "6" jest mylona z cyfrą "0" w 7 przypadkach.
Cyfra "9" jest mylona z cyfrą "4" w 11 przypadkach.
"""

""" d)
Dokładność rośnie zarówno dla zbioru treningowego, jak i walidacyjnego.
Funkcja kosztu maleje zarówno dla zbioru treningowego, jak i walidacyjnego. 
Krzywe są zbliżone i utrzymują podobny trend, co sugeruje stabilność modelu i brak przeuczenia
"""

""" e)
# Model checkpoint to save the model only when validation accuracy improves
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)
"""
