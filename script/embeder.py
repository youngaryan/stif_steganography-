import cv2 as cv
import numpy as np
from typing import List, Tuple
import json
import os

from utilis import show_image

class WatermarkEmbedder:
    def __init__(self, carrier_image_path:str="images/che.png", watermark_image_path:str="images/watermark.png", segment_size:int=5, 
                 carrier_rotate_angle:int=1, carrier_scale_x:float= 1.5, carrier_scale_y:float= 1.5, carrier_crop: Tuple[int,int,int,int]=None,
                 keypoints_metadata=None):
        

        self.carrier_image_path:str=carrier_image_path
        self.watermark_image_path:str=watermark_image_path

        if not self.carrier_image_path.lower().endswith('.png') and  not self.carrier_image_path.lower().endswith('.tif'):
            raise ValueError(f"carrier image must be in .png or .tif format Got: {self.carrier_image_path}")

        if not self.watermark_image_path.lower().endswith('.png') and  not self.watermark_image_path.lower().endswith('.tif'):
            raise ValueError(f"watermark image must be in .png or .tif format Got: {self.watermark_image_path}")



        self.carrier_rotate_angle:int=carrier_rotate_angle
        self.carrier_scale_x:float=carrier_scale_x
        self.carrier_scale_y:float=carrier_scale_y
        self.carrier_crop:Tuple[int,int,int,int]=carrier_crop 
        self.segment_size:int=segment_size if segment_size%2!=0 else segment_size+1 # to make sure there is a center pixel
        # self.error_tolerance = error_tolerance
        
        self.modified_carrier_image_path = None
       

        self.carrier_image_colour:cv.Mat=cv.imread(self.carrier_image_path, cv.IMREAD_COLOR)
        self.carrier_image_grey:cv.Mat=cv.imread(self.carrier_image_path, cv.IMREAD_GRAYSCALE)

        self.watermark_image, self.watermark_image_list=self._fetch_watermark_image()
        

        if keypoints_metadata :
            with open(keypoints_metadata, 'r') as f:
                meta =json.load(f)
                self.kps = [cv.KeyPoint(kp[0], kp[1], kp[2]) for kp in meta['keypoints']]
                self.order = meta['order']
        else:
            self.carrier_image_keypoints:Tuple[cv.KeyPoint]=self.detect_key_points_stif(img=self.carrier_image_grey)
        
        self.used_keypoints:List[cv.KeyPoint] = []


        # self.modified_carrier_image:cv.Mat=self.embed_watermarks()

        # Rotate the modified carrier image 
        # self.modified_carrier_image_rotated = self._rotate_carrier_image(self.carrier_rotate_angle)
        # scale the modified carrier image
        # self.modified_carrier_image_scaled:cv.Mat = self._scale_img(self.modified_carrier_image, self.carrier_scale_x, self.carrier_scale_y)
        # crop the modified image
        # self.modified_carrier_image_cropped:cv.Mat =self._crop_img(self.modified_carrier_image, carrier_crop)
        

    def embed(self, out_path:str="res/embeded_watermatks.png", meta_path:str = "res/meta_data.json"):
        self.modified_carrier_image = self.embed_watermarks(out_path, meta_path)
        self.modified_carrier_image_rotated = self._rotate_carrier_image(self.carrier_rotate_angle)
        self.modified_carrier_image_scaled = self._scale_img(self.modified_carrier_image, self.carrier_scale_x, self.carrier_scale_y)
        self.modified_carrier_image_cropped = self._crop_img(self.modified_carrier_image, self.carrier_crop)

        cv.imwrite(filename=f"{out_path[:-4]}_roated.png", img=self.modified_carrier_image_rotated)
        cv.imwrite(filename=f"{out_path[:-4]}_scaled.png", img=self.modified_carrier_image_scaled)
        cv.imwrite(filename=f"{out_path[:-4]}_cropped.png", img=self.modified_carrier_image_cropped)
        return self.modified_carrier_image
    
        
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
    
    def embed_watermarks(self, out_path:str="res/embeded_watermatks.png", meta_path:str="res/meta_data.json")->cv.Mat:
        carrier_img_copy = self.carrier_image_colour.copy()
        channel:int=0 #blue?
        # for one keypint now

        half_sgement=self.segment_size//2 #to capture the square around the keypoints
        height,width = carrier_img_copy.shape[:2]


        self.meta = {'keypoints': [(kp.pt[0], kp.pt[1], kp.size) for kp in self.carrier_image_keypoints],
                    'segment_size':self.segment_size ,
                    'channel':channel}
        with open(meta_path, 'w') as f:
            json.dump(self.meta, f)



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
                    original_pixel = carrier_img_copy[y, x,channel]
                    carrier_img_copy[y,x,channel]  = (original_pixel & ~1) | bit
        
        self.modified_carrier_image_path = out_path
        cv.imwrite(out_path, carrier_img_copy)
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
    
    def extract_watermark(self, suspect_carrier_img:str="res/embeded_watermatks.png", meta_path:str = "res/meta_data.json")->cv.Mat:
        # if self.used_keypoints == []:
        #     raise RuntimeError("no keypoints has been used, there is nothing to extract")

        if not os.path.exists(suspect_carrier_img):
            raise FileNotFoundError(f"couldn't find {suspect_carrier_img}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"couldn't find {meta_path}")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)




        kps = [cv.KeyPoint(kp[0], kp[1], kp[2]) for kp in meta['keypoints']]
        self.segment_size = meta['segment_size']
        channel = meta['channel']


        suspect_colour = cv.imread(suspect_carrier_img, cv.IMREAD_COLOR)
        suspect = suspect_colour[:,:,channel]
        
        
        height, width = suspect.shape[:2]
        extracted_waterwork:List[np.ndarray] = []
        half_segment=self.segment_size//2

        for keypoint in kps:
            x_keypoint, y_keypoint = int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))

            segment = np.zeros((self.segment_size, self.segment_size), dtype=np.uint8)
            for dy in range(-half_segment,half_segment+1):
                for dx in range(-half_segment, half_segment+1):
                    x = x_keypoint+dx
                    y= y_keypoint+dy
                    if not (0 <= x <width  and 0 <= y < height):
                        continue

                    pixel = suspect[y, x]

                    bit =pixel&1
                    segment[dy + half_segment, dx + half_segment] =0

                    if bit!=0:
                        segment[dy + half_segment, dx + half_segment] = 255

            extracted_waterwork.append(segment)
            

        return extracted_waterwork


    def varify_watermark(self,img_path:str= "res/embeded_watermatks.png", meta_path: str = "res/meta_data.json", error_tollernce = 0.05)->bool:
            extracted_watermark= self.extract_watermark(img_path, meta_path=meta_path)
            reconstructed_watermark =self.reconstruct_full_watermark(extracted_watermark) 
            
            if reconstructed_watermark.shape != self.watermark_image.shape: ##avoid shpe error
                return False
            
            diff = np.abs(reconstructed_watermark.astype(np.float32) -self.watermark_image.astype(np.float32))
            max_diff = 255.0  # assuming 8-bit grayscale
            mismatch_fraction = np.sum(diff >(error_tollernce * max_diff))/diff.size

            return mismatch_fraction <=error_tollernce
    def _rotate_carrier_image(self, angle:int=90)->cv.Mat:

        center_imng= (self.modified_carrier_image.shape[1]//2, self.modified_carrier_image.shape[0]// 2) 
        rotation_matrix = cv.getRotationMatrix2D(center_imng, angle, 1.0) 
        carrier_image_rotated =cv.warpAffine(self.modified_carrier_image, rotation_matrix, (self.modified_carrier_image.shape[1], self.modified_carrier_image.shape[0]))
        return carrier_image_rotated 
    
    def _scale_img(self, img:cv.Mat, scale_x:float = 3.0, scale_y:float = 3.0)->cv.Mat:
        return cv.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)

    def _crop_img(self, img, carrier_crop):

        hieght  ,width = self.modified_carrier_image.shape[:2]
        if carrier_crop:
            x1,x2,y1,y2 = carrier_crop
        else:
            crop_w = int(width* 0.6)
            crop_h = int(hieght *0.6)
            x1 = (width- crop_w)//2 
            y1 = (hieght -crop_h)//2 
            x2 = x1+crop_w
            y2 = y1 +crop_h
            self.carrier_crop = (x1,x2,y1,y2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(hieght, y2)

        
        return self.modified_carrier_image[y1:y2, x1:x2]
    def show_image(self, image_type:int=0):
        """
        show the image using OpenCV.
        :param image_type: 0 for carrier image, 1 for watermark image, 2 for modifed carrier image, 3 for carrier image with keypoints, 4 for extracted waterwork, 
        5, for rotated version of the carrier image, 
        6 for extracted waterwork for rotated carrier image
        7, scaled modefied carried
        8, extracted watermark from scaled carrier,
        9, show cropped carried modeifd image,
        10, shows extracted watermark img from cropped image
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
            extracted_watermark = self.extract_watermark("res/embeded_watermatks.png")
            full_watermark = self.reconstruct_full_watermark(extracted_watermark)
            show_image(full_watermark, title="Reconstructed Full Watermark")
        elif image_type==5:
            show_image(self.modified_carrier_image_rotated, title="Rotated Carrier Image")
        # elif image_type==6:
        #     extracted_watermark = self.extract_watermark("res/embeded_watermatks.png")
        #     full_watermark = self.reconstruct_full_watermark(extracted_watermark)
        #     show_image(full_watermark, title="Reconstructed Full Watermark from Rotated Carrier Image")
        elif image_type==7:
            show_image(self.modified_carrier_image_scaled, "scaled carrier img")
        # elif image_type==8:
        #     extracted_watermark = self.extract_watermark(self.modified_carrier_image_scaled)
        #     full_watermark = self.reconstruct_full_watermark(extracted_watermark)
        #     show_image(full_watermark, title="Reconstructed Full Watermark from scaled Carrier Image")
        elif image_type==9:
            show_image(self.modified_carrier_image_cropped, "scaled carrier img")
        # elif image_type==10:
        #     extracted_watermark = self.extract_watermark(self.modified_carrier_image_cropped)
        #     full_watermark = self.reconstruct_full_watermark(extracted_watermark)
        #     show_image(full_watermark, title="Reconstructed Full Watermark from scaled Carrier Image")
        else:
            raise ValueError("Invalid image type. Use 0 for carrier image or 1 for watermark image.")
        
        cv.waitKey(0)
        cv.destroyAllWindows()