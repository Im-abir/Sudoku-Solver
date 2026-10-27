
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from uuid import uuid4
import os
import cv2
import numpy as np
import tensorflow as tf

from src.grid_detection import find_grid_corners, warp_grid
from src.cell_extraction import split_cells
from src.backtracking_solver import solve


# --------------------------------------------------
# Flask setup
# --------------------------------------------------

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# --------------------------------------------------
# Load CNN model
# --------------------------------------------------

MODEL_PATH = "models/digit_classifier.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "empty"
]


# --------------------------------------------------
# Error page helper
# --------------------------------------------------

def show_error(message, status=200):
    return render_template(
        "index.html",
        error_message=message
    ), status


@app.errorhandler(413)
def file_too_large(error):
    return show_error(
        "This image is too large. Please upload an image "
        "smaller than 10 MB.",
        413
    )


# --------------------------------------------------
# CNN prediction
# --------------------------------------------------

def predict_cell(cell_img, model, class_names, size=28):

    _, thresh_cell = cv2.threshold(
        cell_img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    resized = cv2.resize(
        thresh_cell,
        (size, size)
    )

    normalized = resized.astype("float32") / 255.0

    input_tensor = normalized.reshape(
        1,
        size,
        size,
        1
    )

    pred = model.predict(
        input_tensor,
        verbose=0
    )

    class_idx = int(np.argmax(pred))
    confidence = float(np.max(pred))

    label = class_names[class_idx]

    return label, confidence


# --------------------------------------------------
# Recognize Sudoku
# --------------------------------------------------

def recognize_sudoku(warped):

    warped_gray = cv2.cvtColor(
        warped,
        cv2.COLOR_BGR2GRAY
    )

    cells = split_cells(warped_gray)

    grid = []
    confidences = []

    for row in range(9):

        grid_row = []
        conf_row = []

        for col in range(9):

            cell = cells[row * 9 + col]

            label, confidence = predict_cell(
                cell,
                model,
                class_names
            )

            digit = (
                0 if label == "empty" else int(label)
            )

            grid_row.append(digit)
            conf_row.append(confidence)

        grid.append(grid_row)
        confidences.append(conf_row)

    return np.array(grid), np.array(confidences)


# --------------------------------------------------
# Overlay solution on warped Sudoku
# --------------------------------------------------

def overlay_solution(
    warped_img,
    original_grid,
    solved_grid,
    cell_size=None
):

    overlay = warped_img.copy()

    if cell_size is None:
        cell_size = warped_img.shape[0] // 9

    for row in range(9):

        for col in range(9):

            if original_grid[row][col] == 0:

                digit = solved_grid[row][col]

                x = col * cell_size + cell_size // 3
                y = row * cell_size + int(cell_size * 0.7)

                cv2.putText(
                    overlay,
                    str(digit),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 128, 0),
                    2,
                    cv2.LINE_AA
                )

    return overlay


# --------------------------------------------------
# Unwarp solution onto original image
# --------------------------------------------------

def unwarp_overlay(
    original_img,
    overlay_img,
    M
):

    h, w = original_img.shape[:2]

    M_inv = np.linalg.inv(M)

    unwarped_overlay = cv2.warpPerspective(
        overlay_img,
        M_inv,
        (w, h)
    )

    mask = cv2.warpPerspective(
        np.ones_like(overlay_img) * 255,
        M_inv,
        (w, h)
    )

    mask_gray = cv2.cvtColor(
        mask,
        cv2.COLOR_BGR2GRAY
    )

    _, mask_binary = cv2.threshold(
        mask_gray,
        1,
        255,
        cv2.THRESH_BINARY
    )

    final = original_img.copy()

    final[mask_binary > 0] = (
        unwarped_overlay[mask_binary > 0]
    )

    return final


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")


# --------------------------------------------------
# Solve uploaded Sudoku
# --------------------------------------------------

@app.route("/solve", methods=["POST"])
def solve_sudoku():

    # Check uploaded file

    if "file" not in request.files:
        return show_error(
            "No image was uploaded. Please select a Sudoku image.",
            400
        )

    file = request.files["file"]

    if not file.filename:
        return show_error(
            "Please choose a Sudoku image before clicking Solve.",
            400
        )

    # Validate file extension

    original_filename = secure_filename(file.filename)

    if (
        "." not in original_filename
        or original_filename.rsplit(".", 1)[1].lower()
        not in ALLOWED_EXTENSIONS
    ):
        return show_error(
            "Unsupported file format. Please upload a JPG, "
            "PNG, or WEBP image.",
            400
        )

    # Unique filename prevents overwriting earlier uploads

    extension = original_filename.rsplit(".", 1)[1].lower()

    filename = f"{uuid4().hex}.{extension}"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(input_path)

    # Read image

    image = cv2.imread(input_path)

    if image is None:
        return show_error(
            "We couldn't read this image. Please upload a "
            "clear, valid image file.",
            400
        )

    # Detect Sudoku grid

    corners = find_grid_corners(image)

    if corners is None:
        return show_error(
            "We couldn't detect a Sudoku grid. Please upload "
            "a clearer, well-lit image showing the complete "
            "9×9 puzzle.",
            422
        )

    # Warp Sudoku

    try:
        warped, M = warp_grid(
            image,
            corners,
            size=900
        )

        # Recognize digits

        original_grid, confidences = recognize_sudoku(warped)

    except (cv2.error, ValueError, IndexError) as error:
        app.logger.warning("Image processing failed: %s", error)

        return show_error(
            "We couldn't process the Sudoku image correctly. "
            "Please upload a clearer, straight image of the "
            "complete puzzle.",
            422
        )

    # Print recognition details

    print("\nRecognized Sudoku:")

    for row in original_grid:
        print(row.tolist())

    print(
        "\nTotal recognized digits:",
        np.count_nonzero(original_grid)
    )

    print(
        "Average confidence:",
        np.mean(confidences)
    )

    # Solve Sudoku

    solved_grid = original_grid.copy()

    success = solve(solved_grid)

    if not success:
        return show_error(
            "We couldn't find a solution for this puzzle. "
            "Some digits may have been recognized incorrectly, "
            "the puzzle may be invalid, or the solver may have "
            "reached its search limit. Try a clearer image.",
            422
        )

    print("\nSolved Sudoku:")

    for row in solved_grid:
        print(row.tolist())

    # Overlay solution

    cell_size = warped.shape[0] // 9

    result_warped = overlay_solution(
        warped,
        original_grid,
        solved_grid,
        cell_size
    )

    # Unwarp solution

    result = unwarp_overlay(
        image,
        result_warped,
        M
    )

    # Save unique result

    result_filename = f"solved_{uuid4().hex}.png"

    result_path = os.path.join(
        RESULT_FOLDER,
        result_filename
    )

    if not cv2.imwrite(result_path, result):
        app.logger.error("Could not save result: %s", result_path)

        return show_error(
            "The Sudoku was solved, but the result image "
            "could not be saved. Please try again.",
            500
        )

    # Display successful result

    return render_template(
        "index.html",
        result_image=result_filename,
        original_grid=original_grid.tolist(),
        solved_grid=solved_grid.tolist()
    )


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
