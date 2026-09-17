import cv2
import numpy as np


# ---------------------------------------------------------
# REGION MASKING
# ---------------------------------------------------------

def apply_road_mask(edge_image, polygon):
    """
    Keeps only the pixels inside the specified road polygon.
    """

    mask = np.zeros_like(edge_image)

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    selected_area = cv2.bitwise_and(
        edge_image,
        mask
    )

    return selected_area, mask


# ---------------------------------------------------------
# LINE VISUALIZATION
# ---------------------------------------------------------

def overlay_detected_lines(base_image, detected_lines,
                            line_color=(255, 0, 0),
                            line_width=3):
    """
    Creates a transparent layer containing the detected
    lane segments and blends it with the original frame.
    """

    overlay = np.zeros(
        (
            base_image.shape[0],
            base_image.shape[1],
            3
        ),
        dtype=np.uint8
    )

    if detected_lines is not None:

        for segment in detected_lines:

            for coordinates in segment:

                x_start, y_start, x_end, y_end = coordinates

                cv2.line(
                    overlay,
                    (x_start, y_start),
                    (x_end, y_end),
                    line_color,
                    line_width
                )

    combined = cv2.addWeighted(
        base_image,
        0.8,
        overlay,
        1.0,
        0
    )

    return combined


# ---------------------------------------------------------
# LANE DETECTION
# ---------------------------------------------------------

def detect_lanes(frame):
    """
    Performs lane detection using classical Computer Vision.

    Processing:
        1. Grayscale conversion
        2. Gaussian smoothing
        3. Canny edge detection
        4. Road-region masking
        5. Probabilistic Hough transform
        6. Lane visualization

    Returns:
        annotated frame,
        detected Hough lines,
        intermediate processing images
    """

    img_h, img_w = frame.shape[:2]

    # -----------------------------------------------------
    # GRAYSCALE
    # -----------------------------------------------------

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # GAUSSIAN SMOOTHING
    # -----------------------------------------------------

    smoothed = cv2.GaussianBlur(
        gray_frame,
        (5, 5),
        0
    )

    # -----------------------------------------------------
    # EDGE EXTRACTION
    # -----------------------------------------------------

    edge_map = cv2.Canny(
        smoothed,
        50,
        150
    )

    # -----------------------------------------------------
    # ROAD REGION DEFINITION
    # -----------------------------------------------------

    road_polygon = np.array(
        [[
            (0, img_h),
            (img_w // 2, int(img_h / 2 + 50)),
            (img_w, img_h)
        ]],
        dtype=np.int32
    )

    # -----------------------------------------------------
    # APPLY ROI
    # -----------------------------------------------------

    road_edges, roi_visual = apply_road_mask(
        edge_map,
        road_polygon
    )

    # -----------------------------------------------------
    # HOUGH TRANSFORM
    # -----------------------------------------------------

    detected_lines = cv2.HoughLinesP(
        road_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100
    )

    # -----------------------------------------------------
    # DRAW DETECTED LANE SEGMENTS
    # -----------------------------------------------------

    output_frame = overlay_detected_lines(
        frame.copy(),
        detected_lines
    )

    # Draw ROI boundary for visualization
    cv2.polylines(
        output_frame,
        road_polygon,
        True,
        (255, 0, 255),
        2
    )

    # -----------------------------------------------------
    # SAVE INTERMEDIATE PROCESSING RESULTS
    # -----------------------------------------------------

    processing_stages = {
        "gray": gray_frame,
        "blur": smoothed,
        "edges": edge_map,
        "roi_mask": roi_visual,
        "cropped_edges": road_edges
    }

    return (
        output_frame,
        detected_lines,
        processing_stages
    )

