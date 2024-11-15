import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

print("Current working directory:", os.getcwd())
# Ścieżka do katalogu z danymi
data_dir = "./dogs-cats-mini"

# Parametry wejściowe
img_width, img_height = 128, 128
batch_size = 32

# Generatory danych
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,   # Normalizacja
    validation_split=0.2,  # Podział na zbiór treningowy i walidacyjny
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),  # Wyjście binarne (0 - kot, 1 - pies)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# Trenowanie modelu
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # Liczba epok
    verbose=1,
)

print(f"Liczba próbek w zbiorze treningowym: {train_generator.samples}")
print(f"Liczba próbek w zbiorze walidacyjnym: {val_generator.samples}")

# Wizualizacja dokładności
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Krzywa uczenia się')
plt.show()


# Przewidywanie na zbiorze walidacyjnym
val_generator.reset()
y_true = val_generator.classes
y_pred = (model.predict(val_generator) > 0.5).astype(int)

# Macierz błędu i raport klasyfikacji
print("Macierz błędu:")
print(confusion_matrix(y_true, y_pred))
print("\nRaport klasyfikacji:")
print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))


# Zapis najlepszego modelu
model.save('best_dog_cat_classifier.h5')
print("Model zapisano jako best_dog_cat_classifier.h5")


