import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import SGD

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # a) Co robi StandardScaler? Jak transformowane są dane liczbowe? ->
                                    #      normalizuje dane, tak aby (w przypadku irysów) nie było dużych rozbieżności w wielkości statystyk kwiatów. 
                                    #      Jeśli przeskaluje się je do zakresu [0, 1] to łatwiej będzie nauczyć sieć rozpoznawać gatunki (mniejsza wariancja)

# Encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))     # b) Czym jest OneHotEncoder (i kodowanie „one hot” ogólnie)? Jak etykiety klas są transformowane przez ten encoder? ->
                                                        #       To technika do zamiany danych nienumerycznych na numeryczne, poprzez utworzenie binarnego wektora z liczbą elementów,
                                                        #       równą liczbie kategorii (dla nas 3 gatunki)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])                                                         # c) Model ma 4 warstwy: wejściową, dwie ukryte warstwy z 64 neuronami każda i warstwę wyjściową. Ile 
                                                           # neuronów ma warstwa wejściowa i co oznacza X_train.shape[1]? Ile neuronów ma warstwa wyjściowa i co oznacza y_encoded.shape[1]? ->
                                                           #        Warstwa wejściowa ma tyle neuronów, ile jest kategorii danych (X_train.shape[1]), w naszym przypadku 4. Wyjściowa ma 3 neurony, bo mamy 3 gatunki (y_encoded.shape[1])


""" e) dodanie metryki AUC i zmiana optimizera na SGD sprawiła, że model uczy się szybciej, ale ostatecznie odnosi nieco gorsze wyniki
SGD (Stochastic Gradient Descent) -> prostszy, ale wymaga odpowiedniego doboru szybkości uczenia,
RMSprop -> często stosowany w problemach z sekwencjami,
Adagrad -> dobrze radzi sobie z rzadkimi danymi.
"""
# Compile the model
model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy', AUC()])


""" f) 
Mniejsze batch_size lepiej nadaje się do małych danych, ponieważ model uczy się bardziej szczegółowo, widać większe wahania na wykresach
Większe batch_size może prowadzić do szybszego uczenia w jednej epoce, ale wymaga więcej epok dla zbieżności.
"""
# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=4)


# Evaluate the model on the test set
#test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#print(f"Test Accuracy: {test_accuracy*100:.2f}%")
results = model.evaluate(X_test, y_test, verbose=0)
test_loss, test_accuracy, test_auc = results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test AUC: {test_auc:.4f}")


# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

""" g)
Model osiąga bardzo wysoką dokładność zarówno dla zbioru treningowego, jak i walidacyjnego około 95-98%. Obie krzywe dokładności stabilizują się 
na wysokim poziomie po około 20-30 epokach i pozostają stabilne.
Wartość straty maleje systematycznie w trakcie trenowania, co sugeruje poprawę w dopasowaniu modelu. Zarówno strata treningowa, jak i walidacyjna 
spadają, jednak wartość straty walidacyjnej jest nieco wyższa od straty treningowej po około 10-20 epokach, co jest typowe.
Nie widzimy znaczącego rozjechania się krzywych dokładności i straty pomiędzy zbiorem treningowym a walidacyjnym, co sugeruje, że model dobrze 
generalizuje na zbiorze walidacyjnym.
Brak niedouczenia - bardzo wysoka dokładność zarówno na zbiorze treningowym, jak i walidacyjnym.
"""


""" d) Tahn - podobna do sigmoid, ale skaluje wartości do przedziału [-1, 1], co może pomóc. W naszym przypadku zauważalna różnica jest w poprawności walidacji.

from tensorflow.keras.layers import LeakyReLU

# Define the model with a different activation function
model = Sequential([
    Dense(64, activation='tanh', input_shape=(X_train.shape[1],)),  # tanh zamiast relu
    Dense(64, activation='tanh'),                                   # tanh zamiast relu
    Dense(y_encoded.shape[1], activation='softmax')                 # softmax pozostaje bez zmian
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy with Tanh: {test_accuracy*100:.2f}%")

# Plot the learning curves (analogicznie do poprzedniego kodu)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy with Tanh')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss with Tanh')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()


"""
