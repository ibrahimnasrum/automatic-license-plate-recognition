import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time

class ALPRVideoSystem:
    def __init__(self, model_path, use_gpu=False):
        print("[INFO] Loading YOLOv11 model...")
        self.detector = YOLO(model_path)
        
        print("[INFO] Loading OCR engine...")
        # Force CPU to prevent VRAM crash on MX230
        self.reader = easyocr.Reader(['en'], gpu=use_gpu) 

    def clean_text(self, text):
        """Force-format text to Malaysian standard"""
        alphanumeric = "".join([c for c in text if c.isalnum()])
        
        split_idx = -1
        for i, char in enumerate(alphanumeric):
            if char.isdigit():
                split_idx = i
                break
        
        if split_idx == -1: return text 

        letters = alphanumeric[:split_idx]
        numbers = alphanumeric[split_idx:]

        if len(numbers) > 4 and numbers.endswith('1'):
            numbers = numbers[:-1] 
            
        return f"{letters} {numbers}"

    def preprocess_plate(self, plate_img):
        """
        Preprocessing V3: Includes Morphological Open to separate connected letters.
        """
        # 1. Resize 
        img = cv2.resize(plate_img, (0,0), fx=2, fy=2) 
            
        # 2. Grayscale & Contrast
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = cv2.equalizeHist(gray)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Invert Check (White Text -> Black Text)
        white_pixels = np.sum(binary == 255)
        if white_pixels < (binary.size * 0.5):
            binary = cv2.bitwise_not(binary)
            
        # 4. MORPHOLOGICAL OPEN (The "Scissor" Fix)
        # cuts thin lines connecting letters to the frame/noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 5. Border
        binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        
        return binary

    def unwarp_plate(self, plate_img):
        """
        Smart IPM: Returns clean_warped and best_box contour
        """
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 150) # Looser threshold for video

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        best_box = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True) # 0.04 is robust for video

            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                # Filter for Wide Rectangles (Plates)
                if 2.0 < aspect_ratio < 6.0:
                    best_box = approx
                    break

        # Fallback to original if IPM fails
        if best_box is None:
            return plate_img, None

        # --- WARPING ---
        pts = best_box.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        width, height = 400, 120
        dst = np.array([[0,0], [width-1,0], [width-1,height-1], [0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(plate_img, M, (width, height))
        
        # --- TUNED CROP (The fix for "Phantom 7") ---
        # Top: 2px (keep letters), Sides: 15px (remove frame)
        clean_warped = warped[2:height-10, 15:width-15]

        return clean_warped, best_box

    def run_camera(self, source=0):
        # 1. Start Video Capture
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera source: {source}")
            return

        print("[INFO] Starting video stream... Press 'q' to exit.")
        
        # Set resolution (optional, helps fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape
            LANE_SPLIT_X = 0.5
            split_pixel = int(w * LANE_SPLIT_X)
            
            # Draw Lane Line
            cv2.line(frame, (split_pixel, 0), (split_pixel, h), (255, 0, 0), 2)

            # 1. DETECT (Run YOLO)
            results = self.detector(frame, conf=0.15, verbose=False)
            
            # 2. FILTER CANDIDATES
            candidates = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    
                    # Lane Check (Right side only)
                    if center_x > split_pixel: 
                        area = (x2 - x1) * (y2 - y1)
                        # Save y1 for sorting!
                        candidates.append({'box': box, 'y1': y1, 'area': area})

            # 3. PROCESS THE BEST CANDIDATE
            if candidates:
                # --- SORT BY LOWEST POSITION (Fixes Logo 'G' Issue) ---
                # We want the box closest to the bottom (Highest Y value)
                candidates.sort(key=lambda x: x['y1'], reverse=True)
                
                # Take the lowest one
                best_cand = candidates[0] 
                box = best_cand['box']
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                plate_crop = frame[y1:y2, x1:x2]

                # --- IPM ---
                points_found = None
                try:
                    clean_plate, points = self.unwarp_plate(plate_crop)
                    if points is not None:
                        points_found = points
                except:
                    clean_plate = plate_crop

                # Display IPM Window
                cv2.imshow("Innovation: IPM Result", clean_plate)

                # --- OCR ---
                processed_plate = self.preprocess_plate(clean_plate)
                cv2.imshow("Debug: What OCR Sees", processed_plate)
                
                text_results = self.reader.readtext(processed_plate, detail=0, paragraph=True)
                raw_text = " ".join(text_results).upper()
                final_text = self.clean_text(raw_text)

                # --- VISUALIZATION ---
                # Draw Green Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw IPM Polygon
                if points_found is not None:
                    shifted = points_found.copy()
                    shifted[:, 0, 0] += x1
                    shifted[:, 0, 1] += y1
                    cv2.drawContours(frame, [shifted], -1, (0, 0, 255), 2)
                
                # Draw Text (Only if valid length)
                if len(final_text) >= 3:
                    cv2.putText(frame, final_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    print(f"[LIVE] Detected: {final_text}")

            # Show Main Frame
            # Resize slightly to fit screen
            display_frame = cv2.resize(frame, (1024, int(1024 * h / w)))
            cv2.imshow("ALPR Live Feed", display_frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# --- MAIN ---
if __name__ == "__main__":
    alpr = ALPRVideoSystem(model_path=r"D:\Machine Vision\MV_Project\best.pt", use_gpu=True)
    
    # Use 0 for Webcam, or put a video file path here
    alpr.run_camera(source=0)