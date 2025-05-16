import cv2 as cv
import numpy as np
from typing import List, Tuple
import json
import os
import logging
from pathlib import Path
from utilis import ensure_odd, watermark_to_segements, make_dir



class WatermarkVeryfier:
    def __init__(self, img_path:str, meta_oath:str,watermark_path:str, accetance_thresh:float=0.9)->None:
        self.img_path=Path(img_path)
        self.meta_oath=Path(meta_oath)
        self.watermark_path=Path(watermark_path)
        self.thresh:float=accetance_thresh

        self.meta_data=None

        with open(self.meta_oath,"r")as f:
            self.meta_data = json.load(f)
        
        if  self.meta_data is None:
            raise FileNotFoundError(f"couldn't find the metada at {meta_oath}")
        
        self.segment_size = self.meta_data["segment_size"]
        self.channel = self.meta_data["channel"]
        self.rows = self.meta_data["segement_rows"]
        self.cols = self.meta_data["segment_cols"]


        self.carrier_colour = cv.imread(img_path, cv.IMREAD_COLOR)
        if  self.carrier_colour is None:
            raise FileNotFoundError(f"couldn't find the image at {img_path}")
        self.carrier_grey = cv.imread(img_path, cv.IMREAD_GRAYSCALE)


        self.watermark= cv.imread(watermark_path, cv.IMREAD_GRAYSCALE)
        if  self.watermark is None:
            raise FileNotFoundError(f"couldn't find the image at {watermark_path}")
        
    def verify(self)->Tuple[bool, np.ndarray,List[Tuple[int, int]]]:
        '''return 
            1-is verified
            2- reconstrcted watermarks
            3- mismatch center points
        '''
        sitf=cv.SIFT_create()
        half_segment = self.segment_size//2

        desc_meta=np.asarray([kp["descriptor"] for kp in self.meta_data["keypoints"]], dtype=np.float32)

        keypoints_sus, desc_sus = sitf.detectAndCompute(self.carrier_grey, None)

        if len(keypoints_sus) ==0:
            return False, np.zeros_like(self.watermark), []
        
        #Brute-Force Matcher for Watermark verification, cv2.NORM_L2 for stif
        bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(desc_meta,desc_sus, k=2)

        accepteable_match :List[Tuple[cv.DMatch,int]]=[]#(match,index)

        for idx,(m,n) in enumerate(matches):
            if m.distance< self.thresh*n.distance:
                accepteable_match.append((m,idx))

        if  accepteable_match == []:
            return False, np.zeros_like(self.watermark), []

        reconstructed_watermark = np.zeros_like(self.watermark)
        mismatch_centre:List[Tuple[int, int]] = []     


        for m, idx in accepteable_match:
            dest_kp = keypoints_sus[m.trainIdx]
            x,y =int(round(dest_kp.pt[0])), int(round(dest_kp.pt[1]))
            segment_idx = self.meta_data["keypoints"][idx]["segement_index"]

            segment_bits = np.zeros((self.segment_size, self.segment_size), dtype=np.uint8)
            for dy in range(-half_segment,half_segment+1):
                for dx in range(-half_segment,half_segment+1):
                    bit=self.carrier_colour[y+dy,x+dx,self.channel]&1
                    segment_bits[dy+half_segment,dx+half_segment]=255 if bit else 0
            row,col = segment_idx//self.cols, segment_idx%self.cols

            y_st,x_st=row*self.segment_size,col*self.segment_size
            reconstructed_watermark[y_st:y_st+self.segment_size,x_st:x_st+self.segment_size] = segment_bits
            excepted_segment = self.watermark[y_st:y_st+self.segment_size,x_st:x_st+self.segment_size]

            if not np.array_equal(excepted_segment,segment_bits):
                mismatch_centre.append((x,y))

        auth = len(mismatch_centre)==0
        return auth, reconstructed_watermark,mismatch_centre


