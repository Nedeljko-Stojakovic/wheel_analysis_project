import cv2
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import Dict, Tuple, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WheelAnalyzerCV:
    """
    A class for analyzing wooden wheels in images, detecting spokes and circles.
    """

    def __init__(self, canny_low: int = 50, canny_high: int = 150, hough_param1: int = 50, hough_param2: int = 30,
                 spoke_consistency_threshold: float = 0.6, debug: bool = False):
        """
        Initialize the WheelAnalyzerCV class.

        Args:
            canny_low (int): Lower threshold for Canny edge detection.
            canny_high (int): Upper threshold for Canny edge detection.
            hough_param1 (int): First method-specific parameter for Hough Circle Transform.
            hough_param2 (int): Second method-specific parameter for Hough Circle Transform.
            spoke_consistency_threshold (float): Threshold for considering a spoke as complete.
            debug (bool): Enable debug mode.
        """
        self.script_dir = Path(__file__).parent
        self.results_dir = self.script_dir / '../results'
        self.results_dir.mkdir(exist_ok=True)
        self.debug_dir = self.script_dir / '../debug'
        self.debug_dir.mkdir(exist_ok=True)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.spoke_consistency_threshold = spoke_consistency_threshold
        self.debug = debug

    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed binary image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def detect_wheel(binary: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        Detect the wheel in the binary image.

        Args:
            binary (np.ndarray): Binary image.

        Returns:
            Tuple[Tuple[int, int], int]: Center coordinates and radius of the detected wheel.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wheel_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(wheel_contour)
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius

    def detect_circles(self, binary: np.ndarray, center: Tuple[int, int], radius: int) -> Tuple[int, int]:
        """
        Detect inner and outer circles of the wheel.

        Args:
            binary (np.ndarray): Binary image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            radius (int): Radius of the wheel.

        Returns:
            Tuple[int, int]: Inner and outer radii of the wheel.
        """
        height, width = binary.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)  # type: ignore
        masked_binary = cv2.bitwise_and(binary, binary, mask=mask)

        edges = cv2.Canny(masked_binary, self.canny_low, self.canny_high)

        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=radius // 2,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=int(radius * 0.1),
            maxRadius=int(radius * 0.9)
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            circles = sorted(circles, key=lambda x: x[2])
            outer_circles = [c for c in circles if c[2] < radius * 0.9]
            outer_radius = outer_circles[-1][2] if outer_circles else int(radius * 0.8)
        else:
            outer_radius = int(radius * 0.8)

        inner_radius, central_area = self.detect_inner_circle(masked_binary, center, outer_radius)

        self._debug_circle_detection(masked_binary, center, radius, inner_radius, outer_radius)
        self._save_debug_image(central_area, 'central_area_debug.png')

        return inner_radius, outer_radius

    def detect_inner_circle(self, binary: np.ndarray, center: Tuple[int, int], outer_radius: int) -> (
            Tuple)[int, np.ndarray]:
        """
        Detect the inner circle of the wheel.

        Args:
            binary (np.ndarray): Binary image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            outer_radius (int): Outer radius of the wheel.

        Returns:
            Tuple[int, np.ndarray]: Inner radius and central area of the wheel.
        """
        height, width = binary.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, int(outer_radius * 0.6), 255, -1)  # type: ignore
        central_area = cv2.bitwise_and(binary, binary, mask=mask)

        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(central_area, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=2)

        dist_transform = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
        _, max_val, _, _ = cv2.minMaxLoc(dist_transform)

        inner_radius = int(max_val)
        min_radius = int(outer_radius * 0.1)
        max_radius = int(outer_radius * 0.5)
        inner_radius = max(min_radius, min(inner_radius, max_radius))

        self._debug_inner_circle(dilated, center, inner_radius)

        return inner_radius, central_area

    def detect_spokes(self, binary: np.ndarray, center: Tuple[int, int], inner_radius: int, outer_radius: int) -> (
            Tuple)[List[float], List[float]]:
        """
        Detect complete and broken spokes in the wheel using a radial approach.

        Args:
            binary (np.ndarray): Binary image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            inner_radius (int): Inner radius of the wheel.
            outer_radius (int): Outer radius of the wheel.

        Returns:
            Tuple[List[float], List[float]]: Angles of complete and broken spokes.
        """
        height, width = binary.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, outer_radius, 255, -1)  # type: ignore
        cv2.circle(mask, center, inner_radius, 0, -1)   # type: ignore
        spoke_area = cv2.bitwise_and(binary, binary, mask=mask)

        spoke_area = cv2.equalizeHist(spoke_area)
        spoke_area = cv2.adaptiveThreshold(spoke_area, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        complete_spokes = []
        broken_spokes = []

        num_angles = 360
        min_spoke_length = (outer_radius - inner_radius) * 0.7
        min_broken_length = (outer_radius - inner_radius) * 0.3

        for angle in range(num_angles):
            radial_line = self.get_radial_line(spoke_area, center, inner_radius, outer_radius, angle)
            spoke_profile = self.analyze_radial_line(radial_line)

            if spoke_profile:
                start_radius, end_radius = spoke_profile
                spoke_length = end_radius - start_radius
                if spoke_length >= min_spoke_length:
                    complete_spokes.append(angle)
                elif spoke_length >= min_broken_length:
                    broken_spokes.append(angle)

        complete_spokes = self.merge_spokes(complete_spokes)
        broken_spokes = self.merge_spokes(broken_spokes)

        broken_spokes = [angle for angle in broken_spokes if not any(abs(angle - c_angle) < 5
                                                                     for c_angle in complete_spokes)]

        return complete_spokes, broken_spokes

    @staticmethod
    def get_radial_line(image: np.ndarray, center: Tuple[int, int], inner_radius: int, outer_radius: int,
                        angle: float) -> np.ndarray:
        """
        Extract a radial line from the image.

        Args:
            image (np.ndarray): Input image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            inner_radius (int): Inner radius of the wheel.
            outer_radius (int): Outer radius of the wheel.
            angle (float): Angle of the radial line.

        Returns:
            np.ndarray: Intensity values along the radial line.
        """
        rads = np.radians(angle)
        x = np.round(center[0] + np.cos(rads) * np.arange(inner_radius, outer_radius)).astype(int)
        y = np.round(center[1] + np.sin(rads) * np.arange(inner_radius, outer_radius)).astype(int)
        return image[y, x]

    @staticmethod
    def analyze_radial_line(line: np.ndarray, threshold: int = 128) -> Optional[Tuple[int, int]]:
        """
        Analyze a radial line to find potential spoke.

        Args:
            line (np.ndarray): Intensity values along the radial line.
            threshold (int): Intensity threshold for spoke detection.

        Returns:
            Optional[Tuple[int, int]]: Start and end indices of the detected spoke, if any.
        """
        spoke_start = None
        for i, value in enumerate(line):
            if value > threshold:   # type: ignore
                if spoke_start is None:
                    spoke_start = i
            elif spoke_start is not None:
                return spoke_start, i
        return None

    @staticmethod
    def polar_to_cartesian(center: Tuple[int, int], radius: float, angle: float) -> Tuple[int, int]:
        """
        Convert polar coordinates to cartesian.

        Args:
            center (Tuple[int, int]): Center coordinates.
            radius (float): Radius.
            angle (float): Angle in degrees.

        Returns:
            Tuple[int, int]: Cartesian coordinates (x, y).
        """
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))

        return x, y

    @staticmethod
    def merge_spokes(spokes: List[float], max_angle_diff: float = 20) -> List[float]:
        """
        Merge nearby spokes.

        Args:
            spokes (List[float]): List of spoke angles.
            max_angle_diff (float): Maximum angle difference for merging spokes.

        Returns:
            List[float]: Merged spoke angles.
        """
        if not spokes:
            return []

        merged = []
        current_group = [spokes[0]]

        for angle in spokes[1:]:
            if angle - current_group[-1] <= max_angle_diff:
                current_group.append(angle)
            else:
                merged.append(sum(current_group) / len(current_group))
                current_group = [angle]

        merged.append(sum(current_group) / len(current_group))
        return merged

    def plot_analyzed_wheel(self, image: np.ndarray, center: Tuple[int, int], radius: int, inner_radius: int,
                            outer_radius: int, complete_spokes: List[float], broken_spokes: List[float]) -> plt.Figure:
        """
        Plot the analyzed wheel with detected features.

        Args:
            image (np.ndarray): Input image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            radius (int): Radius of the wheel.
            inner_radius (int): Inner radius of the wheel.
            outer_radius (int): Outer radius of the wheel.
            complete_spokes (List[float]): Angles of complete spokes.
            broken_spokes (List[float]): Angles of broken spokes.

        Returns:
            plt.Figure: Matplotlib figure of the analyzed wheel.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_rgb)

        # Plot center
        ax.plot(center[0], center[1], 'ro', markersize=5)

        # Plot outer radius
        ax.plot([center[0] - outer_radius, center[0] + outer_radius], [center[1], center[1]], color='orange',
                linewidth=2)

        ax.add_artist(plt.Circle(center, radius, fill=False, color='blue', linewidth=2))
        ax.add_artist(plt.Circle(center, inner_radius, fill=False, color='yellow', linewidth=2))
        ax.add_artist(plt.Circle(center, outer_radius, fill=False, color='cyan', linewidth=2))

        for angle in complete_spokes:
            start_point = self.polar_to_cartesian(center, inner_radius, angle)
            end_point = self.polar_to_cartesian(center, outer_radius, angle)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='green', linewidth=2)

        for angle in broken_spokes:
            start_point = self.polar_to_cartesian(center, inner_radius, angle)
            end_point = self.polar_to_cartesian(center, outer_radius, angle)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red', linewidth=2)

        complete_patch = mpatches.Patch(color='green', label='Complete Spokes')
        broken_patch = mpatches.Patch(color='red', label='Broken Spokes')
        wheel_circle = mpatches.Patch(color='blue', label='Wheel Circumference')
        inner_circle = mpatches.Patch(color='yellow', label='Inner Circle')
        outer_circle = mpatches.Patch(color='cyan', label='Outer Circle')
        radius = mpatches.Patch(color='orange', label='Diameter')
        ax.legend(handles=[complete_patch, broken_patch, wheel_circle, inner_circle, outer_circle, radius],
                  loc='upper right')

        ax.set_title("Analyzed Wheel")
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    def _save_debug_image(self, image: np.ndarray, filename: str):
        """Helper method to save debug images with error handling and logging."""
        if not self.debug:
            return
        try:
            output_path = self.debug_dir / filename
            cv2.imwrite(str(output_path), image)
            logging.debug(f"Saved debug image: {output_path}")
        except Exception as e:
            logging.error(f"Failed to save debug image {filename}: {e}")

    def _debug_circle_detection(self, binary: np.ndarray, center: Tuple[int, int], radius: int, inner_radius: int,
                                outer_radius: int):
        """
        Create a debug image for circle detection.

        Args:
            binary (np.ndarray): Binary image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            radius (int): Radius of the wheel.
            inner_radius (int): Inner radius of the wheel.
            outer_radius (int): Outer radius of the wheel.
        """
        if not self.debug:
            return
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, center, radius, (255, 0, 0), 2)
        cv2.circle(debug_img, center, inner_radius, (0, 255, 0), 2)
        cv2.circle(debug_img, center, outer_radius, (0, 0, 255), 2)
        self._save_debug_image(debug_img, 'circle_detection_debug.png')

    def _debug_inner_circle(self, binary: np.ndarray, center: Tuple[int, int], inner_radius: int):
        """
        Create a debug image for inner circle detection.

        Args:
            binary (np.ndarray): Binary image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            inner_radius (int): Inner radius of the wheel.
        """
        if not self.debug:
            return
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, center, inner_radius, (0, 255, 0), 2)
        self._save_debug_image(debug_img, 'inner_circle_debug.png')

    def _debug_spoke_detection(self, binary: np.ndarray, center: Tuple[int, int], inner_radius: int, outer_radius: int,
                               complete_spokes: List[float], broken_spokes: List[float]):
        """
        Create a debug image for spoke detection.

        Args:
            binary (np.ndarray): Binary image.
            center (Tuple[int, int]): Center coordinates of the wheel.
            inner_radius (int): Inner radius of the wheel.
            outer_radius (int): Outer radius of the wheel.
            complete_spokes (List[float]): Angles of complete spokes.
            broken_spokes (List[float]): Angles of broken spokes.
        """
        if not self.debug:
            return
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.circle(debug_img, center, inner_radius, (255, 0, 0), 2)
        cv2.circle(debug_img, center, outer_radius, (255, 0, 0), 2)
        for angle in complete_spokes:
            start_point = self.polar_to_cartesian(center, inner_radius, angle)
            end_point = self.polar_to_cartesian(center, outer_radius, angle)
            cv2.line(debug_img, start_point, end_point, (0, 255, 0), 2)
        for angle in broken_spokes:
            start_point = self.polar_to_cartesian(center, inner_radius, angle)
            end_point = self.polar_to_cartesian(center, outer_radius, angle)
            cv2.line(debug_img, start_point, end_point, (0, 0, 255), 2)
        self._save_debug_image(debug_img, 'spoke_detection_debug.png')

    def analyze_wheel(self, image_path: str) -> Tuple[Dict[str, str], Optional[plt.Figure]]:
        """
        Analyze the wheel in the given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[Dict[str, str], Optional[plt.Figure]]:
            A dictionary containing analysis results and a matplotlib figure (if successful).
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image file: {image_path}")

            binary = self.preprocess_image(image)
            center, radius = self.detect_wheel(binary)
            inner_radius, outer_radius = self.detect_circles(binary, center, radius)
            complete_spokes, broken_spokes = self.detect_spokes(binary, center, inner_radius, outer_radius)

            results = {
                "Total spokes": str(len(complete_spokes) + len(broken_spokes)),
                "Complete spokes": str(len(complete_spokes)),
                "Broken spokes": str(len(broken_spokes)),
                "Wheel diameter": f"{2 * radius} pixels",
                "Wheel center": f"({center[0]}, {center[1]})",
                "Inner radius": f"{inner_radius} pixels",
                "Outer radius": f"{outer_radius} pixels"
            }

            fig = self.plot_analyzed_wheel(image, center, radius, inner_radius, outer_radius, complete_spokes,
                                           broken_spokes)

            input_path = Path(image_path)
            output_filename = f"{input_path.stem}_analyzed{input_path.suffix}"
            output_path = self.results_dir / output_filename
            fig.savefig(str(output_path))
            results["Analyzed image"] = str(output_path)

            return results, fig

        except Exception as e:
            logging.error(f"Error analyzing wheel image {image_path}: {e}")
            return {}, None


def analyze_and_log_image(analyzer, image_path, show_plot):
    """Analyze a single image and log the results."""
    results, fig = analyzer.analyze_wheel(str(image_path))
    logging.info(f"Analysis results for {image_path}:")
    for key, value in results.items():
        logging.info(f"{key}: {value}")

    if show_plot and fig is not None:
        plt.show()


def main():
    """Main function to run the wheel analysis on input images."""
    parser = argparse.ArgumentParser(description="Analyze wooden wheel images for spoke detection.")
    parser.add_argument("--input", default="../data/wooden_wheel_1.png",
                        help="Path to an image file or directory containing images")
    parser.add_argument("--show", action="store_true", help="Show the analyzed image(s)")
    args = parser.parse_args()

    analyzer = WheelAnalyzerCV(debug=True)
    input_path = Path(args.input)

    if input_path.is_file():
        analyze_and_log_image(analyzer, input_path, args.show)
    elif input_path.is_dir():
        for image_file in input_path.glob("*.png"):
            analyze_and_log_image(analyzer, image_file, args.show)
    else:
        logging.error(f"Invalid input path: {input_path}")


if __name__ == "__main__":
    main()
