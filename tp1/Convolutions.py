import numpy as np
import cv2
from matplotlib import pyplot as plt

SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

MASK_SOBEL = {"x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), "y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])}

# Utilisé pour calculer l'indixe d'elément de la matrice
MATRIX_DIFF = np.array([
    (1, -1), (1, 0), (1, 1),
    (0, -1), (0, 0), (0, 1),
    (-1, -1), (-1, 0), (-1, 1)
])


def calculate_gradient(img):
    (h, w) = img.shape
    i_x = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    i_y = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            val_x = np.sum([k * img[y + MATRIX_DIFF[i][0], x + MATRIX_DIFF[i][1]] for i, k in enumerate(MASK_SOBEL["x"])])
            i_x[y, x] = min(max(val_x, 0), 255)

            val_y = np.sum([k * img[y + MATRIX_DIFF[i][0], x + MATRIX_DIFF[i][1]] for i, k in enumerate(MASK_SOBEL["y"])])
            i_y[y, x] = min(max(val_y, 0), 255)

    return np.sqrt(i_x**2 + i_y**2)


def get_image(path):
    """
    Lecture image en niveau de gris et conversion en float64
    """
    return np.float64(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def apply_direct_convolution(img, kernel):
    # Obtenir le largeur et le hateur d'image
    (h, w) = img.shape

    # Applique l'algo
    start_time = cv2.getTickCount()
    img_convolution = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # val = 5 * img[y, x] - img[y - 1, x] - img[y, x - 1] - img[y + 1, x] - img[y, x + 1]
            val = np.sum([k * img[y + MATRIX_DIFF[i][0], x + MATRIX_DIFF[i][1]] for i, k in enumerate(kernel)])
            img_convolution[y, x] = min(max(val, 0), 255)
    end_time = cv2.getTickCount()

    # Calculate le temps d'exécution
    time = (end_time - start_time) / cv2.getTickFrequency()
    print("Méthode directe :", time, "s")

    # Afficher l'image avec la convolution
    cv2.imshow('Avec boucle python',
               img_convolution.astype(np.uint8))  # Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
    cv2.waitKey(0)


def apply_cv_convolution(img, kernel):
    # Applique l'algo
    start_time = cv2.getTickCount()
    img_convolution = cv2.filter2D(img, -1, kernel)
    end_time = cv2.getTickCount()

    # Calculate le temps d'exécution
    time = (end_time - start_time) / cv2.getTickFrequency()
    print("Méthode filter2D :", time, "s")

    # Afficher l'image avec la convolution
    cv2.imshow('Avec filter2D',
               img_convolution / 255.0)  # Convention OpenCV : une image de type flottant est interprétée dans [0,1]
    cv2.waitKey(0)


def apply_direct_mask_sobel(img):
    start_time = cv2.getTickCount()
    gradient = calculate_gradient(img)
    end_time = cv2.getTickCount()

    # Calculate le temps d'exécution
    time = (end_time - start_time) / cv2.getTickFrequency()
    title = "Mask Sobel (direct method)"
    print(title, " - execution time:", time, "s")

    # Afficher l'image avec la convolution
    cv2.imshow(title, gradient.astype(np.uint8))
    cv2.waitKey(0)


image = get_image(path='Image_Pairs/FlowerGarden2.png')
(h, w) = image.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")

apply_direct_convolution(image, kernel=SHARPEN_KERNEL)
apply_cv_convolution(image, kernel=SHARPEN_KERNEL)

apply_direct_mask_sobel(image)
