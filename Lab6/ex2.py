import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale_average(image):
    return np.round(np.mean(image, axis=2)).astype(np.uint8)

def grayscale_weighted(image):
    return (0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]).astype(np.uint8)

image = cv2.imread('test2.jpg')

gray_avg = grayscale_average(image)
gray_weighted = grayscale_weighted(image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Oryginalny obraz")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_avg, cmap='gray')
plt.title("Skala szarości - średnia")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray_weighted, cmap='gray')
plt.title("Skala szarości - ważone")
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Średnia jasność (średnia): {np.mean(gray_avg):.2f}")
print(f"Średnia jasność (ważone): {np.mean(gray_weighted):.2f}")
