import cv2
import matplotlib.pyplot as plt

def get_coordinates(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title("Click 4 points: Top-Left -> Top-Right -> Bottom-Left -> Bottom-Right")
    
    # This function waits for you to click 4 times on the plot
    print("Please click the 4 corners of the lane in the popup window...")
    points = plt.ginput(4) 
    
    print("\n--- COPY THESE COORDINATES ---")
    print(points)
    plt.show()

get_coordinates("C:\\Users\\Asus vivobook\\Desktop\\road_sample.jpg")