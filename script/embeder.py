import cv2 as cv
import numpy as np
from typing import List, Tuple

from utilis import show_image

class WatermarkEmbedder:
    def __init__(self, carrier_image_path:str="images/che.png", watermark_image_path:str="images/watermark.png", segment_size:int=5, 
                 carrier_rotate_angle:int=90, carrier_scale:float= 0.5, carrier_crop: Tuple[int,int,int,int]=None):
        self.carrier_image_path:str=carrier_image_path
        self.watermark_image_path:str=watermark_image_path
        self.carrier_rotate_angle:int=carrier_rotate_angle
        self.carrier_scale:float=carrier_scale
        self.carrier_crop:Tuple[int,int,int,int]=carrier_crop
        self.segment_size:int=segment_size if segment_size%2!=0 else segment_size+1 # to make sure there is a center pixel

        self.carrier_image:cv.Mat=self._fetch_carrier_image()
        self.watermark_image, self.watermark_image_list=self._fetch_watermark_image()
        
        self.carrier_image_keypoints:Tuple[cv.KeyPoint]=self.detect_key_points_stif(img=self.carrier_image)
        self.used_keypoints:List[cv.KeyPoint] = []


        self.modified_carrier_image:cv.Mat=self.embed_watermarks()

        # Rotate the modified carrier image 
        self.carrier_image_rotated_inc_watermark = self._rotate_carrier_image(self.carrier_rotate_angle)
           
        
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
        
    #TODO : add return type for this function
    def _fetch_watermark_image(self,):
        '''returns two values, the watermarks image and a list '''
        watermark = cv.imread(self.watermark_image_path,cv.IMREAD_GRAYSCALE)

        if watermark is None:
            raise FileNotFoundError(f"Watermark image not found at ~/{self.watermark_image_path}.")


        hight, width = watermark.shape[:2]
        
        # check if the image too small for the segmant size
        if hight<self.segment_size or width <self.segment_size:
            raise ValueError(f"""watermark image is too small for the segment size {self.segment_size}.
                             the watermark image size is {hight}x{width}""")

        segments:List[np.ndarray] = []
        for col in range(0, hight - self.segment_size + 1, self.segment_size):
            for row in range(0, width - self.segment_size + 1 ,self.segment_size):
                segment = watermark[col:col+self.segment_size, row:row+self.segment_size]
                segments.append(segment)


        return watermark, segments
    
    def detect_key_points_stif(self, img:np.ndarray)->Tuple[cv.KeyPoint]:
        stif= cv.SIFT_create()
        key_points, _ = stif.detectAndCompute(img, None)

        # need to fix the order of the jeypoint as they are being returned arbitrarily
        key_points = list(key_points) # so can be sort easier

        ## needs to sort from top to bottm y and left to right for simmilar y values
        key_points.sort(key=lambda key_pint:(int(round(key_pint.pt[1])), int(round(key_pint.pt[0]))))

        return key_points
    
    def embed_watermarks(self,)->cv.Mat:
        carrier_img_copy = self.carrier_image.copy()

        # for one keypint now

        half_sgement=self.segment_size//2 #to capture the square around the keypoints
        height,width = carrier_img_copy.shape[:2]

        if len(self.watermark_image_list) > len(self.carrier_image_keypoints):
            print(f"Warn:watermark has {len(self.watermark_image_list)} segements, however the carrier image has only { len(self.carrier_image_keypoints)} keypoints, dome segments will not be embeded, this will cause to loose at least {len(self.watermark_image_list)-len(self.carrier_image_keypoints)} segments")

        
        for waterwork_segment, keypoint in zip(self.watermark_image_list, self.carrier_image_keypoints):

            x_keypoint, y_keypoint = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

            if x_keypoint-half_sgement<0 or y_keypoint-half_sgement<0 or x_keypoint+half_sgement>=width or y_keypoint+half_sgement>=height:
                print("Warn:try smaller segment size, skipping this segment")
                continue
            
            ##adding used keypoints to the list
            self.used_keypoints.append(keypoint)

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
    
    def reconstruct_full_watermark(self, segments: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct the full watermark image from extracted segments.
        Assumes segments were extracted in row-major order.
        """
        
        height, width=self.watermark_image.shape[:2]
        height_segment,width_segment= height//self.segment_size, width//self.segment_size
        total_used_segments = width_segment * height_segment

        frame = np.zeros((height,width), dtype=np.uint8)

        for index, segment in enumerate(segments):
            row_idx = index // width_segment
            col_idx = index % width_segment


            if row_idx >= height_segment or col_idx >= width_segment:
                break

            y_start  = row_idx*self.segment_size
            x_start = col_idx*self.segment_size

            frame[y_start:y_start+self.segment_size, x_start:x_start+self.segment_size]=segment

        print(f"total used segments: {total_used_segments}, and total recovered segments: {len(segments)}")

        return frame
    
    def extract_watermark(self,)->cv.Mat:
        if self.used_keypoints == []:
            raise RuntimeError("no keypoints has been used, there is nothing to extract")
        
        extracted_waterwork:List[np.ndarray] = []
        half_segment=self.segment_size//2

        for keypoint in self.used_keypoints:
            x_keypoint, y_keypoint = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

            segment = np.zeros((self.segment_size, self.segment_size), dtype=np.uint8)
            for dy in range(-half_segment,half_segment+1):
                for dx in range(-half_segment, half_segment+1):
                    x = x_keypoint+dx
                    y= y_keypoint+dy
                    
                    pixel = self.modified_carrier_image[y, x]

                    bit =pixel&1
                    segment[dy + half_segment, dx + half_segment] =0

                    if bit!=0:
                        segment[dy + half_segment, dx + half_segment] = 255

            extracted_waterwork.append(segment)
            

        return extracted_waterwork


    def varify_watermark(self,)->bool:
            extracted_watermark= self.extract_watermark()
            reconstructed_watermark =self.reconstruct_full_watermark(extracted_watermark) 
            

            return np.array_equal(reconstructed_watermark, self.watermark_image) 
    def _rotate_carrier_image(self, angle:int=90):

        center_imng= (self.modified_carrier_image.shape[1]//2, self.modified_carrier_image.shape[0]// 2) 
        rotation_matrix = cv.getRotationMatrix2D(center_imng, angle, 1.0) 
        carrier_image_rotated =cv.warpAffine(self.modified_carrier_image, rotation_matrix, (self.modified_carrier_image.shape[1], self.modified_carrier_image.shape[0]))
        return carrier_image_rotated 
    



    def show_image(self, image_type:int=0):
        """
        show the image using OpenCV.
        :param image_type: 0 for carrier image, 1 for watermark image, 2 for modifed carrier image, 3 for carrier image with keypoints, 4 for extracted waterwork, 
        5, for rotated version of the carrier image, 
        6 for extracted waterwork for rotated carrier image
        :param title: Title of the window.
        """
        if image_type == 0:
            show_image(self.carrier_image, title="Carrier image")
        elif image_type == 1:
            show_image(self.watermark_image,title="waterwork image" )
        elif image_type == 2:
            show_image(self.modified_carrier_image,title="watermarked image(modified carrier image)")
        elif image_type == 3:
            image_with_keypoints = cv.drawKeypoints(
            self.carrier_image, self.carrier_image_keypoints, None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            show_image(image_with_keypoints, title="Carrier image with keypoints")
        elif image_type==4:
            extracted_watermark = self.extract_watermark()
            full_watermark = self.reconstruct_full_watermark(extracted_watermark)
            show_image(full_watermark, title="Reconstructed Full Watermark")
        elif image_type==5:
            show_image(self.carrier_image_rotated_inc_watermark, title="Rotated Carrier Image")
        elif image_type==6:
            extracted_watermark = self.extract_watermark()
            full_watermark = self.reconstruct_full_watermark(extracted_watermark)
            show_image(full_watermark, title="Reconstructed Full Watermark from Rotated Carrier Image")
        else:
            raise ValueError("Invalid image type. Use 0 for carrier image or 1 for watermark image.")
        
        cv.waitKey(0)
        cv.destroyAllWindows()