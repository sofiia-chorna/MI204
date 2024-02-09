import numpy as np
import cv2
from matplotlib import pyplot as plt

SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Utilisé pour calculer l'indixe d'elément de la matrice
matrix_diff = np.array([
    (1, -1), (1, 0), (1, 1),
    (0, -1), (0, 0), (0, 1),
    (-1, -1), (-1, 0), (-1, 1)
])


def get_image(path):
    """
    Lecture image en niveau de gris et conversion en float64
    """
    return np.float64(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def apply_direct_method(img, kernel):
    # Obtenir le largeur et le hateur d'image
    (h, w) = img.shape

    # Applique l'algo
    start_time = cv2.getTickCount()
    img_convolution = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # val = 5 * img[y, x] - img[y - 1, x] - img[y, x - 1] - img[y + 1, x] - img[y, x + 1]
            val = np.sum([k * img[y + matrix_diff[i][0], x + matrix_diff[i][1]] for i, k in enumerate(kernel)])
            print(val)
            img_convolution[y, x] = min(max(val, 0), 255)
    end_time = cv2.getTickCount()

    # Calculate le temps d'exécution
    time = (end_time - start_time) / cv2.getTickFrequency()
    print("Méthode directe :", time, "s")

    # Afficher l'image avec la convolution
    cv2.imshow('Avec boucle python',
               img_convolution.astype(np.uint8))  # Convention OpenCV : une image de type entier est interprétée dans {0,...,255}
    cv2.waitKey(0)

    plt.subplot(121)
    plt.imshow(img_convolution, cmap='gray')
    plt.title('Convolution - Méthode Directe')


def apply_filter2D(img, kernel):
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

    plt.subplot(122)
    plt.imshow(img_convolution, cmap='gray', vmin=0.0, vmax=255.0)  # Convention Matplotlib : par défaut, normalise l'histogramme !
    plt.title('Convolution - filter2D')
    plt.show()

    # h_x = ([-1, 0, 1], [-2, 0, 2], [-1, 0, 1])
    # h_y = np.transpose(h_x)
    #
    # img3 = cv2.filter2d(img, -1, h_x)
    # img4 = cv2.filter2d(img, -1, h_y)


image = get_image(path='Image_Pairs/FlowerGarden2.png')
(h, w) = image.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")

apply_direct_method(image, kernel=SHARPEN_KERNEL)
apply_filter2D(image, kernel=SHARPEN_KERNEL)
