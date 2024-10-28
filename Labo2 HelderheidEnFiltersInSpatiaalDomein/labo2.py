import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy
import scipy.ndimage
from matplotlib.widgets import Slider
from typing import Callable


def gamma_transformatie():
	# Gamma(<1: uitsmering van donkere zones, >1: compressie van donkere zones)
	gamma = 0.3
	image = cv2.imread("images/cat.jpg", cv2.IMREAD_GRAYSCALE)

	normalized = image / 255.0
	corrected = np.power(normalized, gamma)
	corrected = np.uint8(corrected * 255)

	show_results(image, corrected)


def histogram_equalization():
	image = cv2.imread("images/lowcontrast.jpg", cv2.IMREAD_GRAYSCALE)

	equalised = cv2.equalizeHist(image)
	show_results(image, equalised)


def histogram_matching():
	reference = cv2.imread("images/gsm.jpg", cv2.IMREAD_GRAYSCALE)
	source = cv2.imread("images/lowcontrast.jpg", cv2.IMREAD_GRAYSCALE)

	src_hist = cv2.calcHist([source], [0], None, [256], [0, 256])
	ref_hist = cv2.calcHist([reference], [0], None, [256], [0, 256])

	src_cdf = src_hist.cumsum()
	ref_cdf = ref_hist.cumsum()

	src_cdf_normalized = src_cdf / src_cdf.max()
	ref_cdf_normalized = ref_cdf / ref_cdf.max()

	lookup_table = np.zeros(256, dtype=np.uint8)
	j = 1
	for i in range(256):
		while j < 256 and ref_cdf_normalized[j] <= src_cdf_normalized[i]:
			j += 1
		lookup_table[i] = j - 1

	result = cv2.LUT(source, lookup_table)
	show_results(reference, result)


def cumulative_histogram(image):
	hist = cv2.calcHist([image], [0], None, [256], [0, 256])
	cumhist = np.cumsum(hist).astype(np.float64)
	cumhist = cumhist / cumhist[-1]
	return cumhist


def show_results(image, result_image, title="Image"):
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
	ax.imshow(image, cmap=plt.get_cmap("gray"))
	ax0.imshow(result_image, cmap=plt.get_cmap("gray"))
	ax1.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
	ax2.plot(cv2.calcHist([result_image], [0], None, [256], [0, 256]))
	ax3.plot(cumulative_histogram(image))
	ax4.plot(cumulative_histogram(result_image))
	plt.show()


def show_results_interactive(function: Callable, values: list[int], title="Image"):
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
	for i, value in enumerate(values):
		fig.subplots_adjust(bottom=0.25)
		slider_axis = fig.add_axes([0.2, 0.2 - (i / 20), 0.65, 0.03])
		sliders[i] = Slider(
			slider_axis, "Slider" + str(i), valmin=1, valmax=21, valinit=value, valstep=2
		)

		def sliders_on_changed(val, i):
			values[i] = val
			#print(*values)
			image, result_image = function(*values)
			plot_data(image, result_image)
			fig.canvas.draw_idle()
		sliders[i].on_changed(lambda val, idx=i : sliders_on_changed(val, idx))
	plt.show()


def averaging_filter(size: int):
	image = cv2.imread("images/lena_spn.jpg", cv2.IMREAD_GRAYSCALE)

	kernel_size = (size, size)
	mask = np.ones(kernel_size, np.float32) / size**2
	filtered_image = cv2.filter2D(image, -1, mask)
	show_results(image, filtered_image, "Averaging filter")


def gausian_filter(size: int):
	"""
	size: (power of 2) - 1
	"""
	image = cv2.imread("images/lena_spn.jpg", cv2.IMREAD_GRAYSCALE)

	kernel_size = (size, size)
	sigma = 2  # Standard deviation for the Gaussian filter
	gaussian_blur = cv2.GaussianBlur(image, kernel_size, sigma)
	show_results(image, gaussian_blur, "Gausian filter")


def laplacian_filter(size: int):
	image = cv2.imread("images/gsm.jpg", cv2.IMREAD_GRAYSCALE)

	mask = cv2.Laplacian(image, cv2.CV_64F, ksize=size)
	mask = cv2.convertScaleAbs(mask)
	blended = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
	show_results(image, mask, "Laplacian filter - Mask")
	show_results(image, blended, "Laplacian filter - Blended")


def sobel_filter(size: int):
	image = cv2.imread("images/gsm.jpg", cv2.IMREAD_GRAYSCALE)

	# Applying Sobel filter in x and y directions
	sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=size)  # Derivative in x-direction
	sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=size)  # Derivative in y-direction

	# Combine both directions
	mask = cv2.magnitude(sobelx, sobely)
	mask = cv2.convertScaleAbs(mask)

	# Blend the original image with the Sobel mask
	blended = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

	# Show the results
	show_results(image, mask, "Sobel filter - Mask")
	show_results(image, blended, "Sobel filter - Blended")


def minimum_filter(size: int):
	image = cv2.imread("images/lena_spn.jpg", cv2.IMREAD_GRAYSCALE)

	filtered = scipy.ndimage.minimum_filter(image, size)
	show_results(image, filtered, "Minimum filter")


def maximum_filter(size: int):
	image = cv2.imread("images/lena_spn.jpg", cv2.IMREAD_GRAYSCALE)

	filtered = scipy.ndimage.maximum_filter(image, size)
	show_results(image, filtered, "Maximum filter")


def median_filter(size: int):
	image = cv2.imread("images/lena_spn.jpg", cv2.IMREAD_GRAYSCALE)

	filtered = cv2.medianBlur(image, ksize=size)
	show_results(image, filtered, "Median filter")


def pencil(median_ksize, laplacian_ksize):
	image = cv2.imread("images/lena_spn.jpg", cv2.IMREAD_GRAYSCALE)

	filtered = cv2.medianBlur(image, ksize=median_ksize)
	mask = cv2.Laplacian(filtered, ddepth=cv2.CV_16S, ksize=laplacian_ksize)
	mask = cv2.convertScaleAbs(mask)
	inverted = 255 - mask
	show_results(image, inverted, "Pencil")
	return [image, inverted]


def main():
	# Opdracht 1
	gamma_transformatie()

	# Opdracht 2
	histogram_equalization()

	# Opdracht 3
	histogram_matching()

	# Opdracht 4
	# averaging_filter(5)
	# gausian_filter(15)
	# laplacian_filter(3)
	# sobel_filter(3)
	# minimum_filter(2)
	# maximum_filter(2)
	# median_filter(5)
	#show_results_interactive(pencil, [7, 3], "Pencil")
	pencil(7, 3)


if __name__ == "__main__":
	main()
