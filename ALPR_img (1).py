import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re

class ALPRSystem:
    def __init__(self, model_path, use_gpu=False):
        print("[INFO] Loading YOLOv11 model...")
        self.detector = YOLO(model_path)
        
        print("[INFO] Loading OCR engine...")
        # Force CPU (gpu=False)
        self.reader = easyocr.Reader(['en'], gpu=use_gpu) 

    def clean_text(self, text):
        """
        Force-format text to Malaysian standard (e.g., 'JJD:98961' -> 'JJD 9896')
        """
        # 1. Remove special characters (keep only A-Z and 0-9)
        alphanumeric = "".join([c for c in text if c.isalnum()])
        
        # 2. Split into Letters and Numbers
        split_idx = -1
        for i, char in enumerate(alphanumeric):
            if char.isdigit():
                split_idx = i
                break
        
        if split_idx == -1: return text # Return raw if no numbers found

        letters = alphanumeric[:split_idx]
        numbers = alphanumeric[split_idx:]

        # 3. Filter Garbage
        # Malaysian plates usually have max 4 digits. 
        if len(numbers) > 4 and numbers.endswith('1'):
            numbers = numbers[:-1] 
            
        return f"{letters} {numbers}"

    def preprocess_plate(self, plate_img):
        """
        Final Tuning: Thinning + Inversion + Border for Truck Plates
        """
        # 1. Resize (Make it huge to separate pixels)
        img = cv2.resize(plate_img, (0,0), fx=2, fy=2) 
            
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Enhance Contrast
        contrast = cv2.equalizeHist(gray)
        
        # 4. Binary Threshold (Otsu)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. invert check (White Text to Black Text)
        white_pixels = np.sum(binary == 255)
        if white_pixels < (binary.size * 0.5):
            binary = cv2.bitwise_not(binary)
            
        # 6. Erosion
        # Eats away edges to thin letters
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        
        # 7. Add a white border
        # Prevents letters touching the edge
        binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        
        return binary

    def unwarp_plate(self, plate_img):
        """
        INNOVATION: Perspective Correction (IPM).
        Returns: (warped_image, contour_points)
        """
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        display_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Look for rectangle (4 points)
            if len(approx) == 4:
                display_cnt = approx
                break

        # Fallback: If no square found, return original + None
        if display_cnt is None:
            return plate_img, None

        # Order points: TL, TR, BR, BL
        pts = display_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Warp to High Resolution (400x120) for better OCR
        width, height = 400, 120
        dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(plate_img, M, (width, height))
        
        # Crop 5px border to remove screws/frame
        clean_warped = warped[5:height-5, 5:width-5]

        return clean_warped, display_cnt

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Could not read: {image_path}")
            return

        h, w, _ = frame.shape
        LANE_SPLIT_X = 0.5  
        DETECT_ON_RIGHT = True 
        
        # Draw lane line
        split_pixel = int(w * LANE_SPLIT_X)
        cv2.line(frame, (split_pixel, 0), (split_pixel, h), (255, 0, 0), 2)

        # 1. Detect
        results = self.detector(frame, conf=0.15, imgsz=1280, verbose=False)
        
        # 2. Filter Candidates
        candidates = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                
                # Lane Check
                is_on_right = center_x > split_pixel
                if DETECT_ON_RIGHT and not is_on_right: continue
                if not DETECT_ON_RIGHT and is_on_right: continue
                
                area = (x2 - x1) * (y2 - y1)
                candidates.append((area, box))

        # 3. Select Best Candidate (Largest)
        if not candidates:
            print("[INFO] No vehicles found in lane.")
            return

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_area, best_box = candidates[0]
        
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        plate_crop = frame[y1:y2, x1:x2]

        # --- INNOVATION: IPM ---
        points_found = None
        try:
            clean_plate, points = self.unwarp_plate(plate_crop)
            if points is not None:
                points_found = points
        except:
            clean_plate = plate_crop 

        cv2.imshow("Innovation: IPM Result", clean_plate)

        # --- PRE-PROCESS & OCR ---
        processed_plate = self.preprocess_plate(clean_plate)
        
        # Show Debug Window
        cv2.imshow("Debug: What OCR Sees", processed_plate)
        
        text_results = self.reader.readtext(processed_plate, detail=0, paragraph=True)
        raw_text = " ".join(text_results).upper()
        
        # --- CLEANUP ---
        final_text = self.clean_text(raw_text)

        text_results = self.reader.readtext(processed_plate, detail=0, paragraph=True)
        raw_text = " ".join(text_results).upper()
        final_text = self.clean_text(raw_text)

        # --- ignore garbage like logo or dirt.
        if len(final_text) < 3:
            print(f"[IGNORED] Result '{final_text}' is too short (likely a logo/noise).")
            return
        
        print(f"[PLATE NUMBER] {final_text}")

        # --- VISUALIZATION ---
        # Draw Main Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw IPM Red Polygon
        if points_found is not None:
            shifted_points = points_found.copy()
            shifted_points[:, 0, 0] += x1 # Offset X
            shifted_points[:, 0, 1] += y1 # Offset Y
            cv2.drawContours(frame, [shifted_points], -1, (0, 0, 255), 2)
            cv2.putText(frame, "IPM Active", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw Result Text
        cv2.putText(frame, final_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Show Result
        display_frame = cv2.resize(frame, (1024, int(1024 * h / w)))
        cv2.imshow("Final ALPR System", display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- RUN IT ---
if __name__ == "__main__":
    # 1. Update with your model path
    alpr = ALPRSystem(model_path=r"D:\Machine Vision\MV_Project\best.pt", use_gpu=True)
    
    # 2. Update with your image path
    # (Use raw string 'r' for Windows paths)
    img_path = r"D:\Machine Vision\MV_Project\random_images_01\20250918.165808.793.J001A1.SL.SD.NaN.NaN.jpg"
    alpr.process_image(img_path)