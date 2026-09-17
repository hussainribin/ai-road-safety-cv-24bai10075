
import cv2


def analyze_lane_position(frame, detected_lines, threshold_offset=50):
    """
    Evaluates the position of the estimated vehicle center
    with respect to the detected road lane.
    """

    img_height, img_width = frame.shape[:2]
    vehicle_x = img_width // 2

    # No lane information available
    if not detected_lines:
        return (
            frame,
            "NO LANE DETECTED",
            (0, 0, 255)
        )

    left_points = []
    right_points = []

    # Classify detected lines according to their slope
    for group in detected_lines:

        for x1, y1, x2, y2 in group:

            horizontal_change = x2 - x1

            if horizontal_change == 0:
                continue

            line_slope = (y2 - y1) / horizontal_change

            if line_slope < -0.5:
                left_points += [x1, x2]

            elif line_slope > 0.5:
                right_points += [x1, x2]

    # Continue only when both lane sides are available
    if not left_points or not right_points:
        return (
            frame,
            "CAUTION: POOR VISIBILITY",
            (0, 255, 255)
        )

    # Estimate the center of each lane boundary
    left_boundary = sum(left_points) / len(left_points)
    right_boundary = sum(right_points) / len(right_points)

    # Calculate estimated lane center
    estimated_lane_center = int(
        (left_boundary + right_boundary) / 2
    )

    # Calculate horizontal displacement
    displacement = abs(
        estimated_lane_center - vehicle_x
    )

    # Determine road-position condition
    if displacement > threshold_offset:
        message = "LANE POSITION WARNING"
        display_color = (0, 165, 255)
    else:
        message = "SAFE / NORMAL"
        display_color = (0, 255, 0)

    # Reference point for visualization
    reference_y = img_height - 50

    # Estimated lane center
    cv2.circle(
        frame,
        (estimated_lane_center, reference_y),
        10,
        display_color,
        -1
    )

    # Vehicle/ego center
    cv2.circle(
        frame,
        (vehicle_x, reference_y),
        10,
        (255, 255, 255),
        -1
    )

    # Display safety status
    cv2.putText(
        frame,
        f"Status: {message}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        display_color,
        2
    )

    return frame, message, display_color