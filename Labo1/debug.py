import numpy as np
import cv2
import matplotlib.pyplot as plt


def change(image1: np.uint8, image2:np.uint8):
    #return cv2.absdiff(image1, image2)
    return image1.astype(np.int8) - image2.astype(np.int8)

def logicimage(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.xlabel("Intensiteit")
    plt.ylabel("Aantal pixels")
    plt.show()

    _, new_image = cv2.threshold(image, 35, 256, cv2.THRESH_BINARY)
    plt.imshow(new_image)
    plt.show()


def main():
    im1 = plt.imread("images/watch1.jpg")
    im2 = plt.imread("images/watch2.jpg")
    print(type(im1[0, 0]))
    changeIm = change(im1, im2)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.axis("off")
    plt.imshow(im1, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.axis("off")
    plt.imshow(im2, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Change")
    plt.axis("off")
    plt.imshow(changeIm, cmap="gray")
    plt.show()
    
    #logicimage(changeIm)


if __name__ == "__main__":
    main()
