from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

model_path = 'dogs_cats_classifier.h5'

model = load_model(model_path)
print("Model został załadowany.")

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    'dogs-cats-mini/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # brak mieszania danych
)

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()  # Zamiana prawdopodobieństw na klasy (0 lub 1)

true_classes = test_generator.classes

conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))
