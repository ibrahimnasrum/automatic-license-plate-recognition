import cv2
import numpy as np
import matplotlib.pyplot as plt

def simple_cam_to_bev(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # 2. Define the "Region of Interest" (ROI) points in the Camera View
    # These points should form a trapezoid on the road (e.g., lane lines).
    # You must tweak these coordinates to fit your specific image resolution!
    # Order: Top-Left, Top-Right, Bottom-Left, Bottom-Right
    h, w = img.shape[:2]
    
    # Example coordinates for a standard 1280x720 dashcam image
    # (Adjust these based on where the road is in YOUR image)
    src_points = np.float32([
        [723, 774],  # Top-Left (near horizon)
        [1068, 784], # Top-Right
        [27, 1168], # Bottom-Left (hood of car)
        [1588, 1181] # Bottom-Right 
    ])

    # 3. Define where these points should go in the Bird's Eye View
    # We want them to form a perfect rectangle (parallel lines)
    dst_points = np.float32([
        [w * 0.25, 0],         # Top-Left (Top of view)
        [w * 0.75, 0],         # Top-Right
        [w * 0.25, h],         # Bottom-Left (Bottom of view)
        [w * 0.75, h]          # Bottom-Right
    ])

    # 4. Compute the Perspective Transformation Matrix (Homography)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 5. Apply the Warp
    bev_img = cv2.warpPerspective(img, matrix, (w, h))

    # --- Visualization ---
    plt.figure(figsize=(10, 5))

    # Draw the points on original image for visualization
    img_with_pts = img.copy()
    cv2.polylines(img_with_pts, [np.int32(src_points)], True, (0, 0, 255), 3)

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_with_pts, cv2.COLOR_BGR2RGB))
    plt.title("Original Camera View (Source Points)")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
    plt.title("Traditional Bird's Eye View (IPM)")
    
    plt.show()

# Run the function
# Replace 'road_sample.jpg' with your actual image filename
simple_cam_to_bev("C:\\Users\\Asus vivobook\\Desktop\\road_sample.jpg")