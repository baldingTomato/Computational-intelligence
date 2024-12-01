import cv2
import numpy as np
import os

def count_birds_advanced(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Zwiększ kontrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray_enhanced = clahe.apply(gray)
    
    # Usuń szumy przez rozmycie
    blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    
    # Wykrywanie krawędzi
    edges = cv2.Canny(blurred, threshold1=10, threshold2=50)
    
    # Zamykanie morfologiczne
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrowanie konturów
    min_area = 0.5  # Zmniejszony próg
    max_area = 5000  # Większy próg
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 3.0:  # Szeroki zakres proporcji
                filtered_contours.append(cnt)
    
    # Rysowanie konturów
    result_image = image.copy()
    cv2.drawContours(result_image, filtered_contours, -1, (0, 255, 0), 2)
    
    # Liczba ptaków
    bird_count = len(filtered_contours)
    
    # Zapisz wynik
    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, result_image)
    
    return bird_count, output_path

def process_images_in_folder(folder_path):
    # Lista wyników
    results = []
    
    # Iteracja przez wszystkie pliki w folderze
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            bird_count, output_path = count_birds_advanced(image_path)
            results.append((filename, bird_count))
            print(f"Przetworzono: {filename}, Liczba ptaków: {bird_count}, Zapisano do: {output_path}")
    
    # Wydrukuj wyniki
    print("\nPodsumowanie:")
    for filename, bird_count in results:
        print(f"{filename}: {bird_count} ptaków")

# Ścieżka do folderu z obrazami
folder_path = "E:\\Python\\Lab6\\birds"

# Uruchom proces
process_images_in_folder(folder_path)
