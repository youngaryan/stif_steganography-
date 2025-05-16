
from interface import Interface


if __name__ =="__main__":

    # embedder = WatermarkEmbedder(watermark_image_path="images/watermark_A_5x5.png", segment_size=5, carrier_image_path="images/che.png")
    
    # # embedder.show_image(0) #Carrier image
    # # embedder.show_image(1) #Watermark image
    # # embedder.show_image(2) #watermarked image(modified carrier image)
    # # embedder.show_image(3) # Carrier image with keypoints
    # # embedder.show_image(4)
    # # embedder.show_image(5) #Rotated Carrier Image
    # # embedder.show_image(6) #watermarked image(modified rotated carrier image)


    # # embedder.show_image(7) #scaled imge
    # # embedder.show_image(8)

    # # embedder.show_image(9) #cropped imge
    # # embedder.show_image(10)

    # # print("varyfy rotation ",embedder.varify_watermark(img=embedder.modified_carrier_image_rotated))
    # # print("varyfy scaled ",embedder.varify_watermark(img=embedder.modified_carrier_image_scaled))
    # # print("varyfy cropped ",embedder.varify_watermark(img=embedder.modified_carrier_image_cropped))
    
    
    # # print("varyfy actual embded ",embedder.varify_watermark(img_path=embedder.modified_carrier_image_path))

    # detector =TemperDetector(embedder,threshold=0.05)

    # #true return
    # print(detector.detect_temper(modified_img_path="images/che.png")['tampered'])


    # #false return
    # print(detector.detect_temper(modified_img_path="res/embeded_watermatks.png")['tampered'])

    ic = Interface()
    ic.mainloop()