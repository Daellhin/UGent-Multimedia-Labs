import numpy as np
from matplotlib import pyplot as plt
import cv2

from cv2.typing import MatLike, Scalar
from numpy.typing import ArrayLike


# -- Opdracht 1 --
def segment_color(filename: str):
    image = cv2.imread(filename)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red(Two areas)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    # Blue
    lower_blue = np.array([100, 70, 50])
    upper_blue = np.array([130, 255, 255])
    # White
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    # Black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Plot
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.manager.set_window_title("Segmented color")
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    masks = {
        "Red": mask_red,
        "Blue": mask_blue,
        "White": mask_white,
        "Black": mask_black,
    }
    for i, (color, mask) in enumerate(masks.items(), 2):
        plt.subplot(3, 2, i + 1)
        plt.imshow(mask, cmap="gray")
        plt.title(f"{color} Mask")
        plt.axis("off")
    plt.show()


# -- Opdracht 2 --
def plot_image(image: ArrayLike, title="Title", legend: list[list[str]] = []):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(title)

    ax.imshow(image)
    [ax.plot([], [], label=entry[0], color=entry[1]) for entry in legend]

    if len(legend):
        ax.legend()

    plt.show()


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


def are_lines_similar(line1, line2, rho_tolerance=10, theta_tolerance=0.1):
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


def hough(filename: str):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 79)
    filtered_lines = filter_similar_lines(lines)

    # draw_lines(filtered_lines, img, (0, 255, 0))

    # Filter bijna parallelle lijnen
    parallel_lines = []
    tolerance = 0.1
    for i in range(len(filtered_lines)):
        for j in range(i + 1, len(filtered_lines)):
            rho1, theta1 = filtered_lines[i]
            rho2, theta2 = filtered_lines[j]
            if abs(theta1 - theta2) < tolerance:
                parallel_lines.append((rho1, theta1))
                parallel_lines.append((rho2, theta2))

    draw_lines(parallel_lines, img, (0, 0, 255))

    plot_image(img, "Hough parallel lines", [["Parallel", "Blue"]])


# -- Opdracht 3 --
def RANSAC(filename: str, num_lines=5, threshold_distance=2, threshold_inliers=100, max_iterations=1000):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_with_lines = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    y_coords, x_coords = np.nonzero(edges)
    points = np.column_stack((x_coords, y_coords))

    colors = plt.cm.rainbow(np.linspace(0, 1, num_lines))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    # Minimale lengte voor een geldige lijn (als percentage van de beeldbreedte)
    min_line_length = img.shape[1] * 0.3  # 30% van de beeldbreedte

    detected_lines = []
    for line_idx in range(num_lines):
        S = 2
        best_inliers = []
        best_line = None
        max_line_score = 0

        if len(points) < 2:
            break

        for _ in range(max_iterations):
            idx = np.random.choice(len(points), S, replace=False)
            point1, point2 = points[idx]
            x1, y1 = point1
            x2, y2 = point2

            if x2 - x1 == 0 and y2 - y1 == 0:
                continue

            distances = np.abs((x2 - x1) * (y1 - points[:, 1]) - (x1 - points[:, 0]) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            inlier_indices = distances < threshold_distance
            inliers = points[inlier_indices]

            if len(inliers) < threshold_inliers:
                continue

            line_vector = np.array([x2 - x1, y2 - y1])
            line_vector = line_vector / np.linalg.norm(line_vector)
            centered_points = inliers - point1
            projections = np.dot(centered_points, line_vector)
            line_length = np.max(projections) - np.min(projections)
            line_score = line_length * len(inliers)

            if line_score > max_line_score and line_length > min_line_length:
                max_line_score = line_score
                best_inliers = inliers
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)
                best_line = (inliers[min_idx], inliers[max_idx])

        if best_line is not None:
            detected_lines.append(best_line)
            pt1, pt2 = best_line
            cv2.line(
                img_with_lines,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                colors[line_idx].tolist(),
                2,
            )

            points = np.array([p for p in points if not np.any(np.all(p == best_inliers, axis=1))])

    fig = plt.figure(figsize=(12, 6))
    fig.canvas.manager.set_window_title("RANSAC power lines")
    plt.subplot(121)
    plt.imshow(img)
    plt.title("Origineel")

    plt.subplot(122)
    plt.imshow(img_with_lines)
    plt.title("Gedetecteerde lijnen")

    plt.tight_layout()
    plt.show()

    return detected_lines


def main():
    # Opdracht 1: Kleursegmentatie
    segment_color("images/traffic1.jpg")
    segment_color("images/traffic2.jpg")

    # Opdracht 2: Lijndetectie
    hough("images/lines.jpg")

    # Opdracht 3: RANSAC
    RANSAC("images/wires.jpg")


if __name__ == "__main__":
    main()
