import cv2 as cv
import matplotlib.pyplot as plt
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



    return None
if __name__ == "__main__":
    carr = _fetch_carrier_image()
    watermark = _fetch_watermark_image(image_path="images/watermark_A_5x5.png")

    keypoints, _ = detect_key_points_stif(carr)

    detect_key_points_stif(watermark)