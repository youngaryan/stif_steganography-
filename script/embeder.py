import cv2 as cv
import matplotlib.pyplot as plt



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


def _fetch_watermark_image(image_path:str="images/watermark.png"):

    watermark = cv.imread(image_path,cv.IMREAD_GRAYSCALE)

    if watermark is None:
        raise FileNotFoundError(f"Watermark image not found at ~/{image_path}.")
    
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


def preprocess_water_mark(watermark:cv.Mat, patch_size:int = 5, keypoints_size:int=1000)->cv.Mat:
    
    # proccessed_watermark = cv.resizeWindow:
    return None
if __name__ == "__main__":
    carr = _fetch_carrier_image()
    watermark = _fetch_watermark_image(image_path="images/watermark_A_5x5.png")

    keypoints, _ = detect_key_points_stif(carr)

    processed_watermark = preprocess_water_mark(watermark, patch_size=5, keypoints_size=len(keypoints))
    detect_key_points_stif(watermark)