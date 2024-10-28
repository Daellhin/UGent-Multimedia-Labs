import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy
import scipy.ndimage
from matplotlib.widgets import Slider
from typing import Callable


def magnitude_spectrum(fshift: np.ndarray):
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    magnitude_spectrum -= np.min(magnitude_spectrum)
    magnitude_spectrum *= 255.0 / np.max(magnitude_spectrum)
    return magnitude_spectrum


def cumulative_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist / cumhist[-1]
    return cumhist


def circle(R: int, M: int, N: int):
    """
    R: straal
    M: rijen
    N: kolommen
    """
    x, y = np.meshgrid(np.linspace(0, N, N), np.linspace(0, M, M), sparse=True)
    cx, cy = [N / 2, M / 2]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= R**2


def show_results_old(image, result_image, title="Image"):
    # Setup figures
    fig = plt.figure(figsize=(20, 15))
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(321)
    ax.set_title("Original Image")
    ax0 = fig.add_subplot(322)
    ax0.set_title("Image after Transformation")
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of Original")
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram after Transformation")
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of Original")
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram after Transformation")
    ax1.set_xlim([0, 255])
    # Plot data
    ax.imshow(image, cmap=plt.get_cmap("gray"))
    ax0.imshow(result_image, cmap=plt.get_cmap("gray"))
    ax1.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
    ax2.plot(cv2.calcHist([result_image], [0], None, [256], [0, 256]))
    ax3.plot(cumulative_histogram(image))
    ax4.plot(cumulative_histogram(result_image))

    plt.show()


def show_results_old_interactive(
    function: Callable, valueList: list[list[int]], title="Image"
):
    """
    valueList: list with list items in format [value,valmin,valmax,step]
    """
    values = [e[0] for e in valueList]
    paramsList = [e[1:4] for e in valueList]
    image, result_image = function(*values)
    # Setup figures
    fig = plt.figure(figsize=(20, 15))
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(321)
    ax.set_title("Original Image")
    ax0 = fig.add_subplot(322)
    ax0.set_title("Image after Transformation")
    ax1 = fig.add_subplot(323)
    ax1.set_title("Histogram of Original")
    ax2 = fig.add_subplot(324, sharex=ax1)
    ax2.set_title("Histogram after Transformation")
    ax3 = fig.add_subplot(325, sharex=ax1)
    ax3.set_title("Cumulative Histogram of Original")
    ax4 = fig.add_subplot(326, sharex=ax1)
    ax4.set_title("Cumulative Histogram after Transformation")
    ax1.set_xlim([0, 255])

    # Plot data
    def plot_data(image, result_image):
        [x.clear() for x in [ax1, ax2, ax3, ax4]]
        ax1.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
        ax2.plot(cv2.calcHist([result_image], [0], None, [256], [0, 256]))
        ax3.plot(cumulative_histogram(image))
        ax4.plot(cumulative_histogram(result_image))
        ax.imshow(image, cmap=plt.get_cmap("gray"))
        ax0.imshow(result_image, cmap=plt.get_cmap("gray"))

    plot_data(image, result_image)
    # Add sliders
    sliders = [0] * len(values)
    for i, (value, params) in enumerate(zip(values, paramsList)):
        fig.subplots_adjust(bottom=0.25)
        slider_axis = fig.add_axes([0.2, 0.2 - (i / 20), 0.65, 0.03])
        sliders[i] = Slider(
            slider_axis,
            "Slider" + str(i),
            valinit=value,
            valmin=params[0],
            valmax=params[1],
            valstep=params[2],
        )

        # Update values when slider value changes
        def sliders_on_changed(val, i):
            values[i] = val
            image, result_image = function(*values)
            plot_data(image, result_image)
            fig.canvas.draw_idle()

        sliders[i].on_changed(lambda val, idx=i: sliders_on_changed(val, idx))

    # Print last values on close
    def on_close(event):
        print(f"Closed {title}: {values}")

    fig.canvas.mpl_connect("close_event", on_close)
    # Show plot
    plt.show()


def show_results(image, fft_image, filtered_fft, filtered_image, title="Image"):
    # Setup figures
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax1 = fig.add_subplot(221)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2 = fig.add_subplot(222)
    ax2.set_title("FFT")
    ax2.axis("off")
    ax3 = fig.add_subplot(223)
    ax3.set_title("Filtered FFT")
    ax3.axis("off")
    ax4 = fig.add_subplot(224)
    ax4.set_title("Filtered Image")
    ax4.axis("off")
    # Plot data
    ax1.imshow(image, cmap="gray")
    ax2.imshow(magnitude_spectrum(fft_image), cmap="gray")
    ax3.imshow(magnitude_spectrum(filtered_fft), cmap="gray")
    ax4.imshow(filtered_image, cmap="gray")
    # Show plot
    plt.tight_layout()
    plt.show()


def show_results_interactive(
    function: Callable, valueList: list[list[int]], title="Image"
):
    """
    valueList: list with list items in format [value,valmin,valmax,step]
    """
    values = [e[0] for e in valueList]
    paramsList = [e[1:4] for e in valueList]
    image, fft_image, filtered_fft, filtered_image = function(*values)
    # Setup figures
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax1 = fig.add_subplot(221)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2 = fig.add_subplot(222)
    ax2.set_title("FFT")
    ax2.axis("off")
    ax3 = fig.add_subplot(223)
    ax3.set_title("Filtered FFT")
    ax3.axis("off")
    ax4 = fig.add_subplot(224)
    ax4.set_title("Filtered Image")
    ax4.axis("off")

    # Plot data
    def plot_data(image, fft_image, filtered_fft, filtered_image):
        [x.clear() for x in [ax1, ax2, ax3, ax4]]
        ax1.imshow(image, cmap="gray")
        ax2.imshow(magnitude_spectrum(fft_image), cmap="gray")
        ax3.imshow(magnitude_spectrum(filtered_fft), cmap="gray")
        ax4.imshow(filtered_image, cmap="gray")

    plot_data(image, fft_image, filtered_fft, filtered_image)
    # Add sliders
    sliders = [0] * len(values)
    for i, (value, params) in enumerate(zip(values, paramsList)):
        fig.subplots_adjust(bottom=0.25)
        slider_axis = fig.add_axes([0.2, 0.2 - (i / 20), 0.65, 0.03])
        sliders[i] = Slider(
            slider_axis,
            "Slider" + str(i),
            valinit=value,
            valmin=params[0],
            valmax=params[1],
            valstep=params[2],
        )

        # Update values when slider value changes
        def sliders_on_changed(val, i):
            values[i] = val
            image, fft_image, filtered_fft, filtered_image = function(*values)
            plot_data(image, fft_image, filtered_fft, filtered_image)
            fig.canvas.draw_idle()

        sliders[i].on_changed(lambda val, idx=i: sliders_on_changed(val, idx))

    # Print last values on close
    def on_close(event):
        print(f"Closed {title}: {values}")

    fig.canvas.mpl_connect("close_event", on_close)
    # Show plot
    plt.tight_layout()
    plt.show()


def fourier_transform_demo(imagePath: str):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    # fourier transform
    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)  # Verwissel diagonale kwadranten
    m_spectrum = magnitude_spectrum(fft_image_shift)

    # inverse fourier transform
    ifshift = np.fft.ifftshift(fft_image_shift)  # Herverwissel diagonale kwadranten
    ifbeeld = np.fft.ifft2(ifshift).real

    ifbeeld_nor = np.uint8(cv2.normalize(ifbeeld, None, 0, 255, cv2.NORM_MINMAX))
    m_spectrum_norm = np.uint8(cv2.normalize(m_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    show_results_old(ifbeeld_nor, m_spectrum_norm, "Fourier Transform")


def ideal_low_pass(radius: int):
    image = cv2.imread("images/campus.jpg", cv2.IMREAD_GRAYSCALE)
    # fourier transform
    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)  # Verwissel diagonale kwadranten
    size = np.shape(fft_image_shift)
    filter = circle(radius, size[0], size[1])
    filtered_fft = fft_image_shift * filter
    ifshift = np.fft.ifftshift(filtered_fft)  # Herverwissel diagonale kwadranten
    filtered_image = np.fft.ifft2(ifshift).real

    # show_results(image, fft_image_shift, filtered_fft, filtered_image)
    return [image, fft_image_shift, filtered_fft, filtered_image]


def ideal_low_pass_mask(radius: int):
    image = cv2.imread("images/campus.jpg", cv2.IMREAD_GRAYSCALE)
    size = np.shape(image)
    filter = np.uint8(circle(radius, size[0], size[1]))
    ifshift = np.fft.ifftshift(filter)  # Herverwissel diagonale kwadranten
    filtered_image = np.fft.ifft2(ifshift).real

    filter_nor = np.uint8(cv2.normalize(filter, None, 0, 255, cv2.NORM_MINMAX))
    filtered_image_norm = np.uint8(
        cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
    )
    # show_results_old(filter_nor, filtered_image_norm)
    return [filter_nor, filtered_image_norm]


def gaussian_low_pass(d0: int):
    image = cv2.imread("images/campus.jpg", cv2.IMREAD_GRAYSCALE)

    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)
    rows, cols = image.shape
    u, v = np.meshgrid(
        np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2)
    )
    d = np.sqrt(u**2 + v**2)
    gaussian_filter = np.exp(-(d**2) / (2 * (d0**2)))
    filtered_fft = fft_image_shift * gaussian_filter

    ifshift = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(ifshift).real

    # show_results(image, fft_image_shift, filtered_fft, filtered_image)
    return [image, fft_image_shift, filtered_fft, filtered_image]


def butterworth_filter(shape, cutoff, order):
    P, Q = shape
    U, V = np.meshgrid(np.arange(P) - P // 2, np.arange(Q) - Q // 2, indexing="ij")
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / cutoff) ** (2 * order))
    return H


def butterworth_low_pass(radius: int, order: int):
    image = cv2.imread("images/campus.jpg", cv2.IMREAD_GRAYSCALE)

    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)
    filter = butterworth_filter(fft_image_shift.shape, radius, order)

    filtered_fft = fft_image_shift * filter
    ifshift = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(ifshift).real

    # show_results(image, fft_image_shift, filtered_fft, filtered_image)
    return [image, fft_image_shift, filtered_fft, filtered_image]


def butterworth_high_pass(radius: int, order: int):
    image = cv2.imread("images/Xraywrist.jpg", cv2.IMREAD_GRAYSCALE)

    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)
    filter = butterworth_filter(fft_image_shift.shape, radius, order)
    high_pass_filter = 1 - filter  # HHP = 1 - HLP

    filtered_fft = fft_image_shift * high_pass_filter
    ifshift = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(ifshift).real

    # show_results(image, fft_image_shift, filtered_fft, filtered_image)
    return [image, fft_image_shift, filtered_fft, filtered_image]


def butterworth_high_frequency_emphasis(radius: int, order: int, a: float, b: float):
    image = cv2.imread("images/Xraywrist.jpg", cv2.IMREAD_GRAYSCALE)

    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)
    filter = butterworth_filter(fft_image_shift.shape, radius, order)
    high_pass_filter = 1 - filter  # HHP = 1 - HLP
    hhfe_filter = (
        a + b * high_pass_filter
    )  # High-frequency emphasis filter: HHFE = a + bHHP

    filtered_fft = fft_image_shift * hhfe_filter
    ifshift = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(ifshift).real

    # show_results(image, fft_image_shift, filtered_fft, filtered_image)
    return [image, fft_image_shift, filtered_fft, filtered_image]


def butterworth_notch_filter_int(shape, D0, u0, v0, order):
    # Maak een meshgrid voor het filter
    rows, cols = shape
    U, V = np.meshgrid(
        np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2)
    )

    # Bereken de afstand van elke punt in het frequentiedomein tot de notch-locaties (u0, v0) en (-u0, -v0)
    D1 = np.sqrt((U - u0) ** 2 + (V - v0) ** 2)
    D2 = np.sqrt((U + u0) ** 2 + (V + v0) ** 2)

    # Butterworth notch-bandstop filter
    H = 1 / (1 + (D0 / D1) ** (2 * order)) * 1 / (1 + (D0 / D2) ** (2 * order))

    return H


def butterworth_notch_filter(D0, u0, v0, order):
    """
    band stop filter
    """
    # Lees de afbeelding in grayscale
    image = cv2.imread("images/oldtv.jpg", cv2.IMREAD_GRAYSCALE)

    # Voer de Fourier-transformatie uit en verschuif de nul-frequentiecomponent naar het centrum
    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)

    # Maak het notch filter
    filter = butterworth_notch_filter_int(fft_image_shift.shape, D0, u0, v0, order)

    # Pas het filter toe op het fft-beeld
    filtered_fft = fft_image_shift * filter

    # Omgekeerde FFT uitvoeren
    ifshift = np.fft.ifftshift(filtered_fft)
    filtered_image = np.fft.ifft2(ifshift).real

    # show_results(image, fft_image_shift, filtered_fft, filtered_image)
    return [image, fft_image_shift, filtered_fft, filtered_image]


def main():
    # Inleiding - De 2-D discrete Fouriertransformatie
    fourier_transform_demo("images/block.jpg")
    fourier_transform_demo("images/disk.jpg")

    # Opdracht 1 - ideaal laagdoorlaatfilter
    show_results_interactive(ideal_low_pass, [[200, 0, 400, 1]], "Ideal Low Pass")
    show_results_old_interactive(
        ideal_low_pass_mask, [[10, 0, 400, 1]], "Ideal Low Pass Mask"
    )

    # Opdracht 2 - Gaussiaans laagdoorlaatfilter
    show_results_interactive(gaussian_low_pass, [[100, 0, 400, 1]], "Gaussian Low Pass")

    # Opdracht 3 - Butterworth laagdoorlaatfilter
    show_results_interactive(
        butterworth_low_pass, [[30, 0, 400, 1], [2, 0, 10, 1]], "Butterworth Low Pass"
    )

    # Opdracht 4 - Hoogdoorlaatfilters
    show_results_interactive(
        butterworth_high_pass,
        [[30, 0, 400, 1], [3, 0, 10, 1]],
        "Butterworth High Pass",
    )
    show_results_interactive(
        butterworth_high_frequency_emphasis,
        [[70, 0, 400, 1], [3, 0, 10, 1], [0.5, 0, 10, 0.1], [2, 0, 10, 0.1]],
        "Butterworth High Frequency Emphasis",
    )

    # Opdracht 5 - Butterworth notch filter
    show_results_interactive(
        butterworth_notch_filter,
        [[10, 0, 400, 1], [30, 0, 400, 1], [11, 0, 400, 1], [2, 0, 10, 1]],
        "Butterworth band stop",
    )


if __name__ == "__main__":
    main()
