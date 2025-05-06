from embeder import WatermarkEmbedder



if __name__ =="__main__":
    embedder = WatermarkEmbedder(watermark_image_path="images/nike_logo.png", segment_size=5, carrier_image_path="images/che.png")
    
    embedder.show_image(0) #Carrier image
    embedder.show_image(1) #Watermark image
    embedder.show_image(2) #watermarked image(modified carrier image)
    embedder.show_image(3) # Carrier image with keypoints
    embedder.show_image(4)
