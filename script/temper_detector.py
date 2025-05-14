import numpy as np
from embeder import WatermarkEmbedder


class TemperDetector:
    def __init__(self, embeder:WatermarkEmbedder, threshold:float= 0.1 ):
        self.embeder=embeder
        self.threshold=threshold
    

    def detect_temper(self, modified_img_path:str = "res/embeded_watermatks.png", meta_data:str="res/meta_data.json"):
        '''return true if the img has been temperred with otherwisue false'''


        extracted_watermark = self.embeder.extract_watermark(
            suspect_carrier_img=modified_img_path,
            meta_path=meta_data
        )

        full_extracted_watermark = self.embeder.reconstruct_full_watermark(extracted_watermark)

        original_watermark, _= self.embeder._fetch_watermark_image()

        if full_extracted_watermark.shape != original_watermark.shape: ##avoid shpe error
            return True #temperd
        
        diff = np.abs(full_extracted_watermark.astype(np.float32) -original_watermark.astype(np.float32))
        mismatches = np.sum(diff > 0)
        total = diff.size
        frac = mismatches / total

        tampered = frac > self.threshold
        return tampered