"""
Autoinjector Pose Estimation Demo

This module provides a complete pipeline for real-time pose estimation of autoinjectors
using YOLO pose models. The system detects autoinjectors (capped/uncapped classes) and
estimates keypoint locations for the front (tip) and back (base) of the device.

Key Features:
    - Real-time inference on webcam or video files
    - Pose estimation with keypoint visualization
    - Configurable confidence and IoU thresholds
    - Multi-device support (CPU, CUDA, MPS)
    - Optional video output recording
    - Interactive visualization with FPS counter

Usage:
    Run from command line:
        python auto_injector_pose_demo.py --model model.pt --source video.mp4

    Or use programmatically:
        from auto_injector_pose_demo import AutoInjectorPoseDemo
        demo = AutoInjectorPoseDemo("model.pt", conf=0.25, device="cuda")
        demo.run(source="input.mp4", save_path="output.mp4")

Requirements:
    - ultralytics (YOLO)
    - opencv-python (cv2)
    - numpy
    - PyTorch (for model inference)

Author: Computer Vision Project Team
License: See project LICENSE file
"""

# Standard library imports
import argparse
import time

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO


class AutoInjectorPoseDemo:
    """
    Real-time pose estimation demo for autoinjector detection using YOLO pose model.

    This class provides a complete pipeline for detecting autoinjectors in images/video,
    estimating their pose (keypoints), and visualizing results in real-time. The model
    detects two object classes (capped/uncapped) and estimates keypoints for the front
    (tip) and back (base) of the autoinjector.

    Attributes:
        KP_FRONT (int): Keypoint index for the front (tip) of the autoinjector.
        KP_BACK (int): Keypoint index for the back (base) of the autoinjector.
        model (YOLO): Loaded YOLO pose estimation model.
        conf (float): Confidence threshold for detections (0.0-1.0).
        iou (float): Intersection over Union threshold for NMS (0.0-1.0).

    Example:
        >>> demo = AutoInjectorPoseDemo("model.pt", conf=0.25, iou=0.5, device="cuda")
        >>> demo.run(source="video.mp4", save_path="output.mp4")
    """

    # Keypoint indices as defined in the model's keypoint schema
    KP_FRONT = 0  # Front tip of the autoinjector
    KP_BACK = 1  # Back base of the autoinjector

    def __init__(self, model_path, conf=0.25, iou=0.5, device=None):
        """
        Initialize the AutoInjectorPoseDemo with a trained YOLO pose model.

        Args:
            model_path (str): Path to the trained YOLO pose model weights file (.pt).
            conf (float, optional): Confidence threshold for object detection.
                Lower values = more detections but more false positives. Defaults to 0.25.
            iou (float, optional): IoU threshold for Non-Maximum Suppression (NMS).
                Higher values = more overlapping boxes allowed. Defaults to 0.5.
            device (str, optional): Device to run inference on. Options: "cpu", "cuda",
                "mps" (Apple Silicon). If None, uses default device. Defaults to None.

        Note:
            The model is loaded with task="pose" to enable keypoint estimation.
            Device selection is important for performance: CUDA for NVIDIA GPUs,
            MPS for Apple Silicon Macs, CPU as fallback.
        """
        # Load YOLO pose model - task="pose" enables keypoint detection
        self.model = YOLO(model_path, task="pose")

        # Override device if specified (useful for forcing CPU/GPU)
        if device:
            self.model.overrides.update({"device": device})

        # Store detection thresholds
        self.conf = conf
        self.iou = iou

        # Log loaded model classes for verification
        print("Loaded model with classes:", self.model.model.names)

    @staticmethod
    def _put_label(frame, text, org):
        """
        Draw text label with a black background for improved readability.

        This utility method renders text on a frame with a semi-transparent black
        background rectangle to ensure the label is visible regardless of the
        underlying image content (important for computer vision visualization).

        Args:
            frame (np.ndarray): BGR image array to draw on (modified in-place).
            text (str): Text string to display.
            org (tuple): (x, y) coordinates for the bottom-left corner of the text.

        Note:
            Uses OpenCV's LINE_AA for anti-aliased text rendering.
            Color scheme: black background (0, 0, 0), white text (255, 255, 255).
        """
        x, y = org

        # Calculate text dimensions to size the background rectangle
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Draw black background rectangle for text readability
        cv2.rectangle(
            frame, (x - 2, y - th - 4), (x + tw + 2, y + baseline), (0, 0, 0), -1
        )

        # Draw white text with anti-aliasing
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_pose(self, frame, kps):
        """
        Visualize keypoint pose on the frame.

        Draws the autoinjector pose by rendering:
        - A line connecting front and back keypoints (if both are visible)
        - Red circle and label for the front (tip) keypoint
        - Blue circle and label for the back (base) keypoint

        Args:
            frame (np.ndarray): BGR image array to draw on (modified in-place).
            kps (np.ndarray): Keypoint array of shape (N, 3) where each row contains
                [x, y, confidence] for a keypoint. Confidence > 0 indicates visibility.

        Note:
            Keypoint visibility is determined by confidence > 0. This allows the model
            to indicate when a keypoint is occluded or not visible in the frame.
        """
        # Extract front and back keypoint coordinates and confidence scores
        xf, yf, cf = kps[self.KP_FRONT]  # Front keypoint: (x, y, confidence)
        xb, yb, cb = kps[self.KP_BACK]  # Back keypoint: (x, y, confidence)

        # Draw connecting line between keypoints if both are visible
        # Green line represents the autoinjector's orientation
        if cf > 0 and cb > 0:
            cv2.line(frame, (int(xb), int(yb)), (int(xf), int(yf)), (0, 255, 0), 2)

        # Draw front keypoint (tip) - Red circle
        if cf > 0:
            cv2.circle(frame, (int(xf), int(yf)), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "front",
                (int(xf) + 6, int(yf) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Draw back keypoint (base) - Blue circle
        if cb > 0:
            cv2.circle(frame, (int(xb), int(yb)), 5, (255, 0, 0), -1)
            cv2.putText(
                frame,
                "back",
                (int(xb) + 6, int(yb) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

    def _process_frame(self, frame):
        """
        Process a single frame through the pose estimation pipeline.

        Performs object detection and pose estimation on the input frame, then
        visualizes the results by drawing bounding boxes, class labels, and keypoints.

        Pipeline steps:
        1. Run YOLO inference to get detections and keypoints
        2. Extract bounding boxes, class IDs, confidence scores, and keypoints
        3. Draw visualizations (boxes, labels, pose) on the frame
        4. Return annotated frame

        Args:
            frame (np.ndarray): Input BGR image frame (H, W, 3).

        Returns:
            np.ndarray: Annotated frame with bounding boxes, labels, and keypoints
                drawn. Original frame is returned if no detections are found.

        Note:
            This method handles the conversion from PyTorch tensors (CPU/GPU) to
            NumPy arrays for OpenCV rendering. Keypoints are only drawn if available
            in the results (pose task requirement).
        """
        # Run inference with configured confidence and IoU thresholds
        # verbose=False suppresses YOLO's default logging output
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)

        # Early return if no detections found
        if not results:
            return frame

        # Extract first (and typically only) result from batch
        res = results[0]

        # Check if keypoints are available (pose estimation results)
        if hasattr(res, "keypoints") and res.keypoints is not None:
            # Convert PyTorch tensors to NumPy arrays for OpenCV
            # Shape: (num_detections, num_keypoints, 3) -> [x, y, confidence]
            kpts = res.keypoints.data.cpu().numpy()

            # Extract bounding boxes in xyxy format (top-left, bottom-right corners)
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

            # Extract class IDs (integer class indices)
            clses = (
                res.boxes.cls.cpu().numpy().astype(int)
                if res.boxes and res.boxes.cls is not None
                else []
            )

            # Extract confidence scores for each detection
            confs = (
                res.boxes.conf.cpu().numpy()
                if res.boxes and res.boxes.conf is not None
                else []
            )

            # Visualize each detected object with its pose
            for i, kp in enumerate(kpts):
                # Default label fallback
                label = "object"

                # Build class-specific label with confidence score
                if i < len(clses):
                    cls_id = clses[i]
                    # Map class ID to human-readable name (e.g., "capped", "uncapped")
                    name = self.model.model.names.get(cls_id, str(cls_id))
                    # Append confidence score if available
                    label = f"{name} {confs[i]:.2f}" if i < len(confs) else name

                # Draw bounding box and label
                if i < len(boxes):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    # Orange bounding box (BGR: 0, 200, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                    # Position label above box, with minimum y-offset for edge cases
                    self._put_label(frame, label, (x1, max(20, y1 - 8)))

                # Draw keypoint pose visualization
                self._draw_pose(frame, kp)

        return frame

    def run(self, source="", save_path=None):
        """
        Run the pose estimation demo on video stream or webcam.

        Main execution loop that:
        - Opens video source (webcam, video file, or image)
        - Processes frames through the pose estimation pipeline
        - Displays real-time results with FPS counter
        - Optionally saves output to video file
        - Handles user input for graceful shutdown

        Args:
            source (str, optional): Video file path or image path. If empty string,
                uses default webcam (index 0). Defaults to "".
            save_path (str, optional): Output video file path to save annotated frames.
                If None, no video is saved. Defaults to None.

        Note:
            - Press 'q' or ESC to exit the demo
            - FPS calculation uses wall-clock time for accurate real-time measurement
            - Video output uses MP4V codec (widely compatible but may need conversion
              for some players; consider using 'XVID' or 'H264' for better compatibility)
            - Webcam detection: empty string "" triggers default camera (index 0)

        Raises:
            SystemExit: Gracefully exits on user interrupt (ESC/q key).
        """
        # Determine if using live webcam feed
        is_live = source.strip() == ""

        # Open video capture: 0 = default webcam, else use provided path
        cap = cv2.VideoCapture(0 if is_live else source)

        # Validate video source is accessible
        if not cap.isOpened():
            print("Cannot open source:", source)
            return

        # Initialize video writer if output path is specified
        writer = None
        if save_path:
            # Use MP4V codec (fourcc code)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Get source FPS (webcam typically returns 0, so we use default)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30  # Default FPS for webcam or when metadata is unavailable

            # Get frame dimensions from source
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize video writer with source properties
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # FPS calculation: track previous frame time
        prev = 0.0

        # Main processing loop
        while True:
            # Read next frame from source
            ok, frame = cap.read()
            if not ok:
                # End of video or read error
                break

            # Process frame through pose estimation pipeline
            frame = self._process_frame(frame)

            # Calculate and display FPS
            now = time.time()
            fps = 1.0 / (now - prev) if prev > 0 else 0.0
            prev = now
            self._put_label(frame, f"FPS {fps:.1f}", (10, 25))

            # Display annotated frame
            cv2.imshow("Autoinjector Pose Demo", frame)

            # Write frame to output video if writer is initialized
            if writer:
                writer.write(frame)

            # Check for exit command (ESC = 27, 'q' key)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        # Cleanup: release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    """
    Command-line entry point for the autoinjector pose estimation demo.

    Parses command-line arguments and initializes the demo pipeline.
    This function provides a user-friendly CLI interface for running the demo
    with customizable parameters.

    Example usage:
        # Run on webcam with default settings
        python auto_injector_pose_demo.py --model model.pt

        # Process video file and save output
        python auto_injector_pose_demo.py --model model.pt --source input.mp4 --save output.mp4

        # Use GPU with custom confidence threshold
        python auto_injector_pose_demo.py --model model.pt --device cuda --conf 0.3

    Note:
        The --model argument is required as the YOLO pose model weights must be
        provided for inference. All other arguments have sensible defaults.
    """
    parser = argparse.ArgumentParser(
        description="Autoinjector pose estimation demo using YOLO pose model. "
        "Detects autoinjectors and estimates keypoints (front/back) in real-time."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO pose model weights file (.pt format)",
    )

    parser.add_argument(
        "--source",
        default="",
        help="Video file path or image path. Leave empty to use default webcam (index 0)",
    )

    parser.add_argument(
        "--save",
        default="",
        help="Optional output video file path to save annotated results",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0.0-1.0). "
        "Lower = more detections but more false positives. Default: 0.25",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for Non-Maximum Suppression (0.0-1.0). "
        "Higher = more overlapping boxes allowed. Default: 0.5",
    )

    parser.add_argument(
        "--device",
        default="",
        help="Device for inference: 'cpu', 'cuda' (NVIDIA GPU), or 'mps' (Apple Silicon). "
        "If empty, uses default device.",
    )

    args = parser.parse_args()

    # Initialize demo with parsed arguments
    demo = AutoInjectorPoseDemo(
        args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
    )

    # Run demo with specified source and optional save path
    demo.run(source=args.source, save_path=args.save if args.save else None)


if __name__ == "__main__":
    main()
