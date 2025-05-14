from embeder import WatermarkEmbedder


class TemperDetector:
    def __init__(self, embeder:WatermarkEmbedder, threshold:float= 0.1 ):
        self.embeder=embeder
        self.threshold=threshold
    

    def detect_temper(self, modified_img_path:str = "res/embeded_watermatks.png", meta_data:str="res/meta_data.json"):
        extracted_watermark = self.embeder.extract_watermark(
            suspect_carrier_img=modified_img_path,
            meta_path=meta_data
        )

        original_watermark = self.embeder._fetch_watermark_image()
        total, recoverd = len(original_watermark), len(extracted_watermark)

        missing_data = abs(total - recoverd)

        wrong_data = 0
        for i, j in zip(original_watermark, extracted_watermark):
            if i != j:
                wrong_data+=1

        temper = ((missing_data + wrong_data) /total) <= self.threshold

        # might need to return more data as map

        return temper