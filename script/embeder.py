import cv2 as cv
import numpy as np
from typing import List, Tuple
import json
import os
import logging
from pathlib import Path
from utilis import ensure_odd, watermark_to_segements, count_segment

logging.basicConfig(level=logging.INFO,format='[%(levelname)s] %(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


from utilis import show_image

class WatermarkEmbedder:
    def __init__(self, carrier_image_path:str="images/che.png", watermark_image_path:str="images/watermark.png", segment_size:int=5, 
                 carrier_rotate_angle:int=1, carrier_scale_x:float= 1.5, carrier_scale_y:float= 1.5, carrier_crop: Tuple[int,int,int,int]=None,
                 keypoints_metadata=None,output_dir:str="res",channel=0,max_keypoints:int=None):
        

        self.carrier_image_path:str=carrier_image_path
        self.watermark_image_path:str=watermark_image_path
        if not self.carrier_image_path.lower().endswith('.png') and  not self.carrier_image_path.lower().endswith('.tif'):
            raise ValueError(f"carrier image must be in .png or .tif format Got: {self.carrier_image_path}")

        if not self.watermark_image_path.lower().endswith('.png') and  not self.watermark_image_path.lower().endswith('.tif'):
            raise ValueError(f"watermark image must be in .png or .tif format Got: {self.watermark_image_path}")

        self.output_dir=output_dir
        self.make_dir(path=self.output_dir)
        
        self.segment_size:int=ensure_odd(segment_size) # to make sure there is a center pixel
        self.channel=channel
        self.max_keypoints=max_keypoints

        self.carrier_image_colour:cv.Mat=cv.imread(self.carrier_image_path, cv.IMREAD_COLOR)
        if not self.carrier_image_colour:
            raise FileNotFoundError(f"no carrier image found at {self.carrier_image_path}")
        self.carrier_image_grey:cv.Mat=cv.imread(self.carrier_image_path, cv.IMREAD_GRAYSCALE)

        self.watermark_imaget=cv.imread(self.watermark_image_path,cv.IMREAD_GRAYSCALE)
        if not self.watermark_imaget:
            raise FileNotFoundError(f"no watermark image found at {self.watermark_image_path}")
        
        self.water_mark_segments, self.rows, self.colunms=watermark_to_segements(self.watermark_imaget,self.segment_size)
        if not self.water_mark_segments:
            raise ValueError(f"watermark is too big for the chosen segemnt size of {self.segment_size}")
        self.water_mark_segments_size=len(self.water_mark_segments)

        self.carrier_rotate_angle:int=carrier_rotate_angle
        self.carrier_scale_x:float=carrier_scale_x
        self.carrier_scale_y:float=carrier_scale_y
        self.carrier_crop:Tuple[int,int,int,int]=carrier_crop 
        
    def make_dir(self, path="res")-> None:
        path_obj = Path(path)
        if path_obj.exists():
            logging.info(f"directory {path} exist.")
        else:
            path_obj.mkdir(parents=True,exist_ok=True)
            logging.info(f"created a directory at {path}.")
    
    def embed(self,)->dict:
        sitf=cv.SIFT_create()
        keypoints,decss=sitf.detectAndCompute(self.carrier_image_grey,None)
        if len(keypoints)==0:
            raise RuntimeError("no keypoints detected, try different carrier image.")
        #sort based on strength of kkeypoints
        sorted_keypoints_desc=sorted(zip(keypoints,decss), key=lambda point:point[0],reverse=True)
       
        max_keypoints_allowed= min(self.water_mark_segments_size,len(sorted_keypoints_desc) )

        #avoid oeverlapping keypoints using occupied mask which is a 2d array the same height and width as our grey carrier filled with false
        selected_keypoints:List[Tuple[cv.KeyPoint, np.ndarray]]=[]
        occupied = np.zeros(self.carrier_image_grey.shape[:2],dtype=bool)
        half_segment = self.segment_size//2
        height,width=self.carrier_image_grey.shape

        for key_point, decs in sorted_keypoints_desc:
            if len(selected_keypoints) > max_keypoints_allowed:
                logging.warn(f"all key points have been filled up, data loss likley")
                break
            x,y=int(round(key_point.pt[0])), int(round(key_point.pt[1]))

            if x-half_segment < 0 or y-half_segment<0 or x+half_segment>=width or y+half_segment>=y:
                continue;

            occup_check =occupied[y-half_segment:y+half_segment+1, x-half_segment:x+half_segment+1]
            if occup_check.any():
                continue
            occupied[y-half_segment:y+half_segment+1, x-half_segment:x+half_segment+1] = True

        if len(selected_keypoints) < self.water_mark_segments_size:
            raise RuntimeError(f"couldn't find enough key points to encode the watermarks. \n found {len(selected_keypoints)} usable keypoing, needs {self.water_mark_segments_size} keypoints")
        # embdeing
        watermarked = self.carrier_image_colour.copy()
        #TODO might need to fix this
        for _,((keypoint, _), segment) in enumerate(zip(selected_keypoints, self.water_mark_segments)):
            x,y=int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))
            for dy in range(-half_segment, half_segment+1):
                for dx in range(-half_segment, half_segment+1):
                    bit=0
                    if segment[dy+half_segment, dx+half_segment] >128:
                        bit=1
                    pix = watermarked[y+dy, x+dx, self.channel]
                    watermarked[y+dy, x+dx, self.channel]=(pix & ~1) | bit
        

        output_path:str =f"{self.output_dir}/{self.carrier_image_path}_watermarked.png"
        cv.imwrite(output_path, watermarked)


        meta_data:dict={
            "segment_size": self.segment,
            "channel": self.channel,
            "segement_rows": self.rows,
            "segment_cols": self.cols,
            "keypoints":[{
                "pt":[float(keypoint.pt[0]), float(keypoint.pt[1])],
                "size":keypoint.size,
                "descriptor":decss.tolist(),
                "segement_index":i,
            }
            for i,(keypoint, decss) in enumerate(selected_keypoints)

            ],
        }

        output_meta_path:str =f"{self.output_dir}/{self.carrier_image_path}_meta.json"

        with open(output_meta_path,"w")as f:
            json.dump(meta_data,f)
        
        return {"img_path":output_path, "meta_path":output_meta_path}

