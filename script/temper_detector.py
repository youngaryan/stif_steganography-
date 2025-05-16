
import cv2 as cv
import numpy as np
from typing import List, Tuple
import json
import os
import logging
from pathlib import Path
from utilis import ensure_odd, watermark_to_segements, make_dir

from watermark_verifer import WatermarkVeryfier
class TamperDetector:
    def __init__(self, img_path:str, meta_oath:str,watermark_path:str,out_dir:str="res" )->None:
        self.img_path=(img_path)
        self.meta_oath=(meta_oath)
        self.watermark_path=(watermark_path)
        self.output_path=Path(out_dir)
        make_dir(path=out_dir,)
    
    def detect(self)->dict:
        verifer = WatermarkVeryfier(
            img_path=self.img_path, meta_oath=self.meta_oath, watermark_path=self.watermark_path
        )

        auth,_,mismatches = verifer.verify()

        tempered_path=None
        if mismatches:
            img = cv.imread(self.img_path)
            for(x,y)in mismatches:
                cv.circle(img, (x,y), 8, (0,0,255), 2)
                tempered_path = f"{Path(self.img_path)}_tempered.png"
                cv.imwrite(tempered_path,img)

        return {
            "tampered":not auth,
            "mismatching_points":(mismatches),
            "overley_tempred_path":tempered_path
        }
        
