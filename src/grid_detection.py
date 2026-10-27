
import cv2
import numpy as np


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    return np.array([
        pts[np.argmin(s)],  # top-left
        pts[np.argmin(d)],  # top-right
        pts[np.argmax(s)],  # bottom-right
        pts[np.argmax(d)]   # bottom-left
    ], dtype=np.float32)


def find_grid_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = img.shape[0] * img.shape[1]

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:100]:
        area = cv2.contourArea(contour)

        if area < image_area * 0.05:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        corners = order_corners(approx)

        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])

        width = (top + bottom) / 2
        height = (left + right) / 2

        if min(width, height) < 1:
            continue

        ratio = width / height

        if 0.65 <= ratio <= 1.5:
            return corners

    return None


def warp_grid(img, corners, size=900):
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        dst
    )

    warped = cv2.warpPerspective(
        img, M, (size, size)
    )

    return warped, M
