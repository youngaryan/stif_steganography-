import cv2 as cv
import matplotlib.pyplot as plt

from typing import List, Tuple
# vonsider making this a class


def _preprocess_water_mark(watermark:cv.Mat, segment_size:int = 5,)->cv.Mat:
    """break down the watermark to smaller segements so it can be embeded in the carrier image"""
    # proccessed_watermark = cv.resizeWindow:
    hight, width = watermark.shape[:2]
    segments = []

    h_crop, w_crop = hight//segment_size, width//segment_size

    for col in range(0, h_crop, segment_size):
        for row in range(0, w_crop,segment_size):
            segment = watermark[col:col+segment_size, row:row+segment_size]
            segments.append(segment)


    return segments

def _fetch_carrier_image(image_path:str="images/che.png") -> cv.Mat:
    """
    Fetch the carrier image from the given path.
    :param image_path: Path to the carrier image.
    :return: Carrier image as a cv.Mat object.
    """
    # Read the image using OpenCV
    carrier_image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if carrier_image is None:
        raise FileNotFoundError(f"Carrier image not found at ~/{image_path}.")
    
    return carrier_image


def _fetch_watermark_image(image_path:str="images/watermark.png", preprocess:bool=True, segment_size=5):

    watermark = cv.imread(image_path,cv.IMREAD_GRAYSCALE)

    if watermark is None:
        raise FileNotFoundError(f"Watermark image not found at ~/{image_path}.")
    

    if preprocess:
        watermark = _preprocess_water_mark(watermark, segment_size=segment_size)
        return watermark


    return watermark

def detect_key_points_stif(image:cv.Mat):
    stif= cv.SIFT_create()

    key_points, descripts = stif.detectAndCompute(image, None)
    # Draw keypoints on the image
    image_with_keypoints = cv.drawKeypoints(
        image, key_points, None,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Display the image
    plt.imshow(image_with_keypoints, cmap='gray')
    plt.title('SIFT Keypoints')
    plt.axis('off')
    plt.show()

    return key_points, descripts



def embed_watermarks(carrier_img:cv.Mat, watermark:List, key_points:cv.KeyPoint, segment_size:int=5)->cv.Mat:

    carrier_img_copy = carrier_img.copy()

    # for one keypint now
    x_keypoints, y_keypoints = int(round(key_points.pt[0])), int(round(key_points.pt[1]))

    half_sgement=segment_size//2 #to capture the square around the keypoints

    if x_keypoints-half_sgement<0 or y_keypoints-half_sgement<0 or x_keypoints+half_sgement>=carrier_img_copy.shape[1] or y_keypoints+half_sgement>=carrier_img_copy.shape[0]:
        print("try smaller segment size or watermark, skipping this segment")
        return carrier_img_copy
    

    for dy in range(-half_sgement,half_sgement+1):
        for dx in range(-half_sgement, half_sgement+1):
            x = x_keypoints+dx
            y= y_keypoints+dy
            bit = 0

            if watermark[dy+half_sgement,dx+half_sgement]>0:
                bit = 1
            original_pixel = carrier_img_copy[y, x]
            carrier_img_copy[y,x]  = (original_pixel & ~1) | bit
    

    return carrier_img_copy



if __name__ == "__main__":
    carr = _fetch_carrier_image()
    watermark = _fetch_watermark_image(image_path="images/watermark_A_5x5.png")

    keypoints, _ = detect_key_points_stif(carr)

    detect_key_points_stif(watermark)