import time

from src.lane_detection import detect_lanes
from src.vehicle_detection import VehicleDetector
from src.lane_analysis import analyze_lane_position


class RoadSafetyPipeline:

    def __init__(self):
        """Initialize the vehicle detection component."""
        self.detector = VehicleDetector()

    def process_frame(self, frame, conf_threshold=0.5):
        """
        Executes the complete road-safety computer vision pipeline.

        Flow:
        Input Frame
            ↓
        Lane Detection & Image Processing
            ↓
        Vehicle Detection
            ↓
        Lane Position Analysis
            ↓
        Final Result
        """

        # Start performance measurement
        pipeline_start = time.perf_counter()

        # =================================================
        # STAGE 1: LANE DETECTION
        # =================================================

        lane_frame, lane_segments, cv_stages = detect_lanes(
            frame
        )

        # =================================================
        # STAGE 2: VEHICLE DETECTION
        # =================================================

        detection_frame, vehicle_total, confidence_average = (
            self.detector.detect(
                lane_frame,
                conf_threshold
            )
        )

        # =================================================
        # STAGE 3: SPATIAL / LANE POSITION ANALYSIS
        # =================================================

        analyzed_frame, safety_status, status_color = (
            analyze_lane_position(
                detection_frame,
                lane_segments
            )
        )

        # =================================================
        # PERFORMANCE CALCULATION
        # =================================================

        elapsed_time = time.perf_counter() - pipeline_start

        # Number of Hough line segments
        segment_count = (
            len(lane_segments)
            if lane_segments is not None
            else 0
        )

        # =================================================
        # FINAL PIPELINE OUTPUT
        # =================================================

        results = {
            "final_image": analyzed_frame,
            "status": safety_status,
            "vehicle_count": vehicle_total,
            "avg_conf": confidence_average,
            "process_time": elapsed_time,
            "intermediates": cv_stages,
            "lines_detected": segment_count
        }

        return results

