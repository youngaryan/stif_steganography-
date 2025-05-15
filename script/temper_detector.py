import numpy as np
from embeder import WatermarkEmbedder


class TemperDetector:
    def __init__(self, embeder:WatermarkEmbedder, threshold:float= 0.05 ):
        self.embeder=embeder
        self.threshold=threshold
    

    def detect_temper(self, modified_img_path:str = "res/embeded_watermatks.png", meta_data:str="res/meta_data.json") ->dict:
        '''return a dict inclduing
            "tampered":tampered,
            "mismatches":int(mismatches),
            "total_pixels":int(total),
            "mismatch_fraction":float(frac),
            "threshold":self.threshold

            and -1 for all values exrpt tampered of shape mismatch
        '''


        extracted_watermark = self.embeder.extract_watermark(
            suspect_carrier_img=modified_img_path,
            meta_path=meta_data
        )

        full_extracted_watermark = self.embeder.reconstruct_full_watermark(extracted_watermark)

        original_watermark, _= self.embeder._fetch_watermark_image()

        if full_extracted_watermark.shape != original_watermark.shape: ##avoid shpe error
            return {
            "tampered": tampered,
            "mismatches": -1,
            "total_pixels": -1,
            "mismatch_fraction": -1,
            "threshold": self.max_mismatch_fraction
        }
        
        diff = np.abs(full_extracted_watermark.astype(np.float32) -original_watermark.astype(np.float32))
        mismatches = np.sum(diff > 0)
        total = diff.size
        frac = mismatches / total

        tampered = frac > self.threshold
        return {
            "tampered": tampered,
            "mismatches": int(mismatches),
            "total_pixels": int(total),
            "mismatch_fraction": float(frac),
            "threshold": self.threshold
        }