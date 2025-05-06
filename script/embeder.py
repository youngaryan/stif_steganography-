import cv2 as cv
from typing import List, Tuple

from utilis import show_image

class WatermarkEmbedder:
    def __init__(self, carrier_image_path:str="images/che.png", watermark_image_path:str="images/watermark.png", segment_size:int=5):
        self.carrier_image_path=carrier_image_path
        self.watermark_image_path=watermark_image_path
        self.segment_size=segment_size
        self.carrier_image=self._fetch_carrier_image()
        self.watermark_image_list=self._fetch_watermark_image(process=True)
        self.watermark_image = self._fetch_watermark_image(process=False)
        self.carrier_image_keypoints=self.detect_key_points_stif()
        
    def _fetch_carrier_image(self,) -> cv.Mat:
        """
        Fetch the carrier image from the given path.
        :param image_path: Path to the carrier image.
        :return: Carrier image as a cv.Mat object.
        """
        # Read the image using OpenCV
        carrier_image = cv.imread(self.carrier_image_path, cv.IMREAD_GRAYSCALE)
        
        # Check if the image was loaded successfully
        if carrier_image is None:
            raise FileNotFoundError(f"Carrier image not found at ~/{self.carrier_image_path}.")
        
        return carrier_image
        
        
    def _fetch_watermark_image(self,process:bool=True)->List[cv.Mat]:

        watermark = cv.imread(self.watermark_image_path,cv.IMREAD_GRAYSCALE)

        if watermark is None:
            raise FileNotFoundError(f"Watermark image not found at ~/{self.watermark_image_path}.")
        
        if not process:
            return watermark

        hight, width = watermark.shape[:2]
        segments = []
        # check if the image too small for the segmant size
        if hight<self.segment_size or width <self.segment_size:
            raise ValueError(f"""watermark image is too small for the segment size {self.segment_size}.
                             the watermark image size is {hight}x{width}""")

        h_crop, w_crop = hight//self.segment_size, width//self.segment_size

        for col in range(0, h_crop, self.segment_size):
            for row in range(0, w_crop,self.segment_size):
                segment = watermark[col:col+self.segment_size, row:row+self.segment_size]
                segments.append(segment)


        return segments
    

    def detect_key_points_stif(self,)->Tuple[cv.KeyPoint]:
        stif= cv.SIFT_create()
        key_points, _ = stif.detectAndCompute(self.carrier_image, None)

        return key_points
    
    def embed_watermarks(self,)->cv.Mat:

        carrier_img_copy = self.carrier_image.copy()

        # for one keypint now

        half_sgement=self.segment_size//2 #to capture the square around the keypoints


        
        for waterwork_segment, keypoint in zip(self.watermark_image_list, self.carrier_image_keypoints):

            x_keypoint, y_keypoint = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

            if x_keypoint-half_sgement<0 or y_keypoint-half_sgement<0 or x_keypoint+half_sgement>=carrier_img_copy.shape[1] or y_keypoint+half_sgement>=carrier_img_copy.shape[0]:
                print("try smaller segment size or watermark, skipping this segment")
                continue

            for dy in range(-half_sgement,half_sgement+1):
                for dx in range(-half_sgement, half_sgement+1):
                    x = x_keypoint+dx
                    y= y_keypoint+dy
                    bit = 0

                    if waterwork_segment[dy+half_sgement,dx+half_sgement]>0:
                        bit = 1
                    original_pixel = carrier_img_copy[y, x]
                    carrier_img_copy[y,x]  = (original_pixel & ~1) | bit
            

        return carrier_img_copy
    
    def show_image(self, image_type:int=0):
        """
        show the image using OpenCV.
        :param image_type: 0 for carrier image, 1 for watermark image, 2 for modifed carrier image, 3 for carrier image with keypoints.
        :param title: Title of the window.
        """
        if image_type == 0:
            show_image(self.carrier_image, title="Carrier image")
        elif image_type == 1:
            show_image(self.watermark_image,title="waterwork image" )
        elif image_type == 2:
            show_image(self.embed_watermarks(),title="watermarked image(modified carrier image)")
        elif image_type == 3:
            image_with_keypoints = cv.drawKeypoints(
            self.carrier_image, self.carrier_image_keypoints, None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            show_image(image_with_keypoints, title="Carrier image with keypoints")
        else:
            raise ValueError("Invalid image type. Use 0 for carrier image or 1 for watermark image.")
        
        cv.waitKey(0)
        cv.destroyAllWindows()
