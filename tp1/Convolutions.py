import numpy as np
import cv2
from matplotlib import pyplot as plt

# Utilisé pour calculer l'indixe d'elément de la matrice
MATRIX_DIFF = np.array([
    (1, -1), (1, 0), (1, 1),
    (0, -1), (0, 0), (0, 1),
    (-1, -1), (-1, 0), (-1, 1)
])

SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
MASK_SOBEL = {"x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), "y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])}


def get_pixel_partial_derivative(img, y, x, kernel):
    partial_derivative = np.sum(
        [k * img[y + MATRIX_DIFF[i][0], x + MATRIX_DIFF[i][1]] for i, k in enumerate(kernel)
    ])
    return min(max(partial_derivative, 0), 255)


def calculate_gradient(img, method="cv"):
    if method not in ["cv", "direct"]:
        ValueError(f"Method {method} is not found. Available options: cv, direct")

    (h, w) = img.shape

    if method == "direct":
        i_x = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        i_y = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                i_x[y, x] = get_pixel_partial_derivative(img, y, x, kernel=MASK_SOBEL["x"])
                i_y[y, x] = get_pixel_partial_derivative(img, y, x, kernel=MASK_SOBEL["y"])
    else:
        i_x = cv2.filter2D(img, -1, MASK_SOBEL["x"])
        i_y = cv2.filter2D(img, -1, MASK_SOBEL["y"])

    return np.sqrt(i_x ** 2 + i_y ** 2)


def get_image(path):
    """
    Lecture image en niveau de gris et conversion en float64
    """
    return np.float64(cv2.imread(path, cv2.IMREAD_GRAYSCALE))


def apply_mask_sobel(img, method="cv"):
    if method not in ["cv", "direct"]:
        ValueError(f"Method {method} is not found. Available options: cv, direct")

    is_method_direct = method == "direct"

    start_time = cv2.getTickCount()
    gradient = calculate_gradient(img, method)
    end_time = cv2.getTickCount()

    # Calculate le temps d'exécution
    time = (end_time - start_time) / cv2.getTickFrequency()

    # Afficher le temps d'exec
    title = "Mask Sobel {}".format("(direct method)" if is_method_direct else "(cv)")
    print(title, " - execution time:", time, "s")

    # Afficher l'image avec la mosk sobel
    cv2.imshow(title, gradient.astype(np.uint8) if is_method_direct else gradient / 255.0)
    cv2.waitKey(0)


def apply_convolution(img, kernel, method="cv"):
    if method not in ["cv", "direct"]:
        ValueError(f"Method {method} is not found. Available options: cv, direct")

    is_method_direct = method == "direct"

    # Obtenir le largeur et le hateur d'image
    (h, w) = img.shape

    # Applique l'algo
    start_time = cv2.getTickCount()
    if is_method_direct:
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = np.sum([k * img[y + MATRIX_DIFF[i][0], x + MATRIX_DIFF[i][1]] for i, k in enumerate(kernel)])
                img[y, x] = min(max(val, 0), 255)
    else:
        img = cv2.filter2D(img, -1, kernel)

    end_time = cv2.getTickCount()

    # Calculate le temps d'exécution
    time = (end_time - start_time) / cv2.getTickFrequency()
    title = "Convolution {}".format("(direct method)" if is_method_direct else "(cv)")
    print(title, " - execution time:", time, "s")

    # Afficher l'image avec la convolution
    cv2.imshow(title, img.astype(np.uint8) if is_method_direct else img / 255.0)
    cv2.waitKey(0)


# Obtenir l'image
image = get_image(path='Image_Pairs/FlowerGarden2.png')
(h, w) = image.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")

# Convolutions
apply_convolution(image, kernel=SHARPEN_KERNEL, method="direct")
apply_convolution(image, kernel=SHARPEN_KERNEL, method="cv")

# Mask sober
apply_mask_sobel(image, method="direct")
apply_mask_sobel(image, method="cv")
