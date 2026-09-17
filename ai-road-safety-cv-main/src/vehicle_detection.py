from ultralytics import YOLO


class VehicleDetector:

    def __init__(self, model_path="yolov8n.pt"):
        """
        Load the YOLO model and define the road-vehicle
        categories that should be detected.
        """

        self.detector = YOLO(model_path)

        # COCO class IDs:
        # 2 = Car
        # 3 = Motorcycle
        # 5 = Bus
        # 7 = Truck
        self.target_categories = (2, 3, 5, 7)

    def detect(self, frame, conf_threshold=0.5):
        """
        Detect selected vehicle categories in the supplied frame.

        Returns:
            processed_frame
            number_of_vehicles
            average_detection_confidence
        """

        # Run object detection
        prediction = self.detector.predict(
            source=frame,
            classes=list(self.target_categories),
            conf=conf_threshold,
            verbose=False
        )

        current_result = prediction[0]

        # Generate YOLO annotated frame
        processed_frame = current_result.plot()

        # Obtain detected bounding boxes
        detected_boxes = current_result.boxes

        if detected_boxes is None or len(detected_boxes) == 0:
            return processed_frame, 0, 0.0

        # Total detected vehicles
        total_vehicles = len(detected_boxes)

        # Extract confidence values
        confidence_values = (
            detected_boxes.conf.detach()
            .cpu()
            .numpy()
        )

        # Mean detection confidence
        mean_confidence = float(
            confidence_values.mean()
        )

        return (
            processed_frame,
            total_vehicles,
            mean_confidence
        )

