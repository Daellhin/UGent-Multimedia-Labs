import math
import statistics
import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy
import scipy.ndimage
from matplotlib.widgets import Slider
from collections import namedtuple

# Typing
import typing
from cv2.typing import MatLike, Scalar
from numpy.typing import ArrayLike
from typing import Any

Line = namedtuple("Line", ["rho", "theta"])


def plot_image(image: ArrayLike, title="Title", legend: list[list[str]] = []):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title)

    ax.imshow(image)
    [ax.plot([], [], label=entry[0], color=entry[1]) for entry in legend]

    if len(legend):
        ax.legend()

    plt.show()


def cumulative_histogram(image: MatLike):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cumhist = np.cumsum(hist).astype(np.float64)
    cumhist = cumhist / cumhist[-1]
    return cumhist


def show_comparison(
    function: callable,
    valueList: list[list[int]],
    title="Image",
    show_histogram=True,
    interactive=True,
):
    """
    valueList: list with list items in format [value,valmin,valmax,step]
    function: returns list of 4 images
    """
    values = [e[0] for e in valueList]
    image, result_image = function(*values)
    # Setup figures
    fig = plt.figure(figsize=(40, 30))
    fig.canvas.manager.set_window_title(title)
    rows = 2 if show_histogram else 1
    ax = fig.add_subplot(rows, 2, 1)
    ax.set_title("Original Image")
    ax0 = fig.add_subplot(rows, 2, 2)
    ax0.set_title("Image after Transformation")

    if show_histogram:
        ax1 = fig.add_subplot(4, 2, 5)
        ax1.set_title("Histogram of Original")
        ax1.set_xlim([0, 255])
        ax2 = fig.add_subplot(4, 2, 6, sharex=ax1)
        ax2.set_title("Histogram after Transformation")
        ax3 = fig.add_subplot(4, 2, 7, sharex=ax1)
        ax3.set_title("Cumulative Histogram of Original")
        ax4 = fig.add_subplot(4, 2, 8, sharex=ax1)
        ax4.set_title("Cumulative Histogram after Transformation")

    # Plot data
    def plot_data(image, result_image):
        ax.imshow(image, cmap=plt.get_cmap("gray"))
        ax0.imshow(result_image, cmap=plt.get_cmap("gray"))
        if show_histogram:
            [x.clear() for x in [ax1, ax2, ax3, ax4]]
            ax1.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
            ax2.plot(cv2.calcHist([result_image], [0], None, [256], [0, 256]))
            ax3.plot(cumulative_histogram(image))
            ax4.plot(cumulative_histogram(result_image))

    plot_data(image, result_image)

    # Add sliders
    if interactive:
        paramsList = [e[1:4] for e in valueList]
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
    if not show_histogram:
        plt.tight_layout(pad=10)
    plt.show()


def plot_image(image: ArrayLike, title="Title", legend: list[list[str]] = []):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title)

    ax.imshow(image)
    [ax.plot([], [], label=entry[0], color=entry[1]) for entry in legend]

    if len(legend):
        ax.legend()

    plt.show()


def create_gaussian_kernel(size=15, sigma=3):
    """Create a 2D Gaussian kernel."""
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def wiener_filter(image_path: str, kernel, k=0.01):
    img = cv2.imread(image_path)

    # Process each channel separately
    restored_channels = []
    for channel in cv2.split(img):
        # Pad the kernel to match image size by placing it in the center
        padded_kernel = np.zeros(channel.shape)
        kh, kw = kernel.shape
        center_y = padded_kernel.shape[0] // 2 - kh // 2
        center_x = padded_kernel.shape[1] // 2 - kw // 2
        padded_kernel[center_y : center_y + kh, center_x : center_x + kw] = kernel

        # Convert to frequency domain
        # H = np.fft.fft2(np.fft.ifftshift(padded_kernel))
        H = np.fft.fft2(np.fft.fftshift(padded_kernel))
        G = np.fft.fft2(channel.astype(float))

        # Apply Wiener filter
        H_conj = np.conj(H)
        H_abs_sq = np.abs(H) ** 2
        F = H_conj / (H_abs_sq + k) * G

        # Convert back to spatial domain
        restored = np.abs(np.fft.ifft2(F))

        # Normalize to [0, 255] range
        restored = (restored - restored.min()) * 255 / (restored.max() - restored.min())
        restored_channels.append(restored.astype(np.uint8))

    # Merge channels back together
    restored_image = cv2.merge(restored_channels)

    # show_results_old(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
    # plot_image(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
    return [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB),
    ]


# -- Opracht 3 --
def draw_lines(lines: list, image: MatLike, color: Scalar):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # print(f"Line coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        cv2.line(image, (x1, y1), (x2, y2), color, 2)


def are_lines_similar(line1: Line, line2: Line, rho_tolerance=10, theta_tolerance=0.1):
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Normaliseer de rho waarden (kan negatief zijn)
    if rho1 < 0:
        rho1 = -rho1
        theta1 = theta1 + np.pi
    if rho2 < 0:
        rho2 = -rho2
        theta2 = theta2 + np.pi

    # Normaliseer theta waarden tussen 0 en pi
    theta1 = theta1 % np.pi
    theta2 = theta2 % np.pi

    theta_diff = min(abs(theta1 - theta2), np.pi - abs(theta1 - theta2))
    if theta_diff > theta_tolerance:
        return False

    rho_diff = abs(rho1 - rho2)
    return rho_diff <= rho_tolerance


def filter_similar_lines(lines: MatLike, rho_tolerance=10, theta_tolerance=0.1):
    if lines is None:
        return []

    filtered_lines = []
    lines = lines[:, 0]

    for line in lines:
        is_similar = False
        for filtered_line in filtered_lines:
            if are_lines_similar(line, filtered_line, rho_tolerance, theta_tolerance):
                is_similar = True
                break

        if not is_similar:
            filtered_lines.append(line)

    return filtered_lines


def hough(
    filename: str,
    canny_lower_threshold=50,
    canny_upper_threshold=200,
    showAllLines=False,
):
    """
    - canny_upper_threshold: Edges with intensity gradient below this value are discarded
    - canny_lower_threshold: Edges with intensity gradient above this value are included in the output
    """
    image = cv2.imread(filename)
    image_with_lines = image.copy()
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        grayscale_image, canny_lower_threshold, canny_upper_threshold, apertureSize=3
    )

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 79)
    filtered_lines = filter_similar_lines(lines)

    if showAllLines:
        draw_lines(filtered_lines, image_with_lines, (0, 255, 0))

    # Filter bijna parallelle lijnen
    parallel_lines: list[Line] = []
    tolerance = 0.1
    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            rho1, theta1 = filtered_lines[i]
            rho2, theta2 = filtered_lines[j]
            if abs(theta1 - theta2) < tolerance:
                parallel_lines.append(Line(rho1, theta1))
                parallel_lines.append(Line(rho2, theta2))

    draw_lines(parallel_lines, image_with_lines, (0, 0, 255))

    Out = namedtuple("Out", ["images", "lines"])
    return Out([image, image_with_lines], parallel_lines)


def detect_motion_direction_hough(
    image_path: str, canny_lower_threshold=50, canny_upper_threshold=200
):
    lines = hough(image_path, canny_lower_threshold, canny_upper_threshold).lines
    return 90 + statistics.fmean(([math.degrees(line.theta) for line in lines]))


def motion_blur_psf(angle: float, length: int):
    psf = np.zeros((length, length))
    cv2.line(psf, (0, length // 2), (length - 1, length // 2), 1, thickness=1)
    rot_matrix = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    psf = cv2.warpAffine(psf, rot_matrix, (length, length))
    return psf / psf.sum()  # Normalize the PSF


def andromeda_callback(k: float, length: int):
    angle = detect_motion_direction_hough(
        "images/Andromeda_motion_blurred_correct.png", 40, 80
    )
    print(angle)
    kernel = motion_blur_psf(180 - angle, length)
    # plot_image(kernel)

    return wiener_filter("images/Andromeda_motion_blurred_correct.png", kernel, k)


def main():
    # Opdracht 1: Wiener filter Japan
    # k=1/10^3
    kernel = create_gaussian_kernel(15, 3)
    plot_image(kernel)
    show_comparison(
        lambda k: wiener_filter("images/Japan_blurred.png", kernel, k),
        [[0.001, 0, 0.01, 0.0001]],
        "Wiener filter Japan",
        interactive=False,
        show_histogram=False,
    )

    # Opdracht 2: Wiener filter Japan Noise
    kernel = create_gaussian_kernel(15, 3)
    show_comparison(
        lambda k: wiener_filter("images/Japan_noisy_blurred.png", kernel, k),
        [[0.01, 0, 0.1, 0.0001]],
        "Wiener filter Japan Noise",
        interactive=False,
        show_histogram=False,
    )

    # Opdracht 3: Wiener filter Andromeda
    # # Helper used to find canny thresholds
    # show_comparison(
    #     lambda a, b: hough("images/Andromeda_motion_blurred_correct.png", a, b).images,
    #     [[40, 0, 400, 1], [80, 0, 800, 1]],
    #     "Hough Andromeda",
    # )

    show_comparison(
        andromeda_callback,
        [[0.001, 0, 0.005, 0.00001], [80, 60, 100, 1]],
        "Wiener filter Andromeda",
        interactive=False,
        show_histogram=False,
    )


if __name__ == "__main__":
    main()
