import numpy as np
import cv2
import matplotlib.pyplot as plt


# functions here ...
def mirror1():
    image = cv2.imread("images/woman_baby.jpg")
    option = 0
    while option != "H" and option != "V" and option != "D":
        option = input(
            "Geef spiegeling(H: horizontaal, V: verticaal, D: diagonaal):"
        ).strip()

    image_flipped = 0
    if option == "H":
        image_flipped = cv2.flip(image, 1)
    elif option == "V":
        image_flipped = cv2.flip(image, 0)
    elif option == "D":
        image_flipped = cv2.transpose(image)

    cv2.imshow("Image", image_flipped)
    cv2.waitKey()
    cv2.destroyAllWindows()


def mirror2():
    image = cv2.imread("images/woman_baby.jpg")
    option = 0
    while option != "H" and option != "V" and option != "D":
        option = input(
            "Geef spiegeling(H: horizontaal, V: verticaal, D: diagonaal):"
        ).strip()

    image_flipped = image
    if option == "H":
        image_flipped = image[:, ::-1]  # reverse x axis
    elif option == "V":
        image_flipped = image[::-1, :]  # reverse y axis
    elif option == "D":
        h, w, c = image.shape
        image_flipped = np.zeros((w, h, c), dtype=image.dtype)

        for i in range(h):
            for j in range(w):
                image_flipped[j, i] = image[i, j]

    cv2.imshow("Image", image_flipped)
    cv2.waitKey()
    cv2.destroyAllWindows()


def swap():
    image = cv2.imread("images/cameraman.jpg")
    h, w, c = image.shape
    h2 = h // 2
    w2 = w // 2

    new_image = image.copy()
    new_image[h2:h, w2:w] = image[0:h2, 0:w2]  # Q2 -> Q4
    new_image[0:h2, 0:w2] = image[h2:h, w2:w]  # Q4 -> Q2
    new_image[h2:h, 0:w2] = image[0:h2, w2:w]  # Q1 -> Q3
    new_image[0:h2, w2:w] = image[h2:h, 0:w2]  # Q3 -> Q1

    cv2.imshow("Image", new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def logicimage():
    image = cv2.imread("images/Xraywrist.jpg")

    plt.hist(image.ravel(), 256, [0, 256])
    plt.xlabel("Intensiteit")
    plt.ylabel("Aantal pixels")
    plt.show()

    _, new_image = cv2.threshold(image, 150, 256, cv2.THRESH_BINARY)
    plt.imshow(new_image)
    plt.show()


def change(image1: np.uint8, image2: np.uint8):
    # return image1 - image2 # original
    return cv2.absdiff(image1, image2)

def main():
    # -- Opdracht 1 --
    mirror1()
    mirror2()

    # -- Opdracht 2 --
    swap()

    # -- Opdracht 3 --
    logicimage()


if __name__ == "__main__":
    main()
