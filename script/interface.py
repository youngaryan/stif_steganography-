import tkinter as tk
from tkinter import filedialog,messagebox

from embeder import WatermarkEmbedder
from watermark_verifer import WatermarkVeryfier
from temper_detector import TamperDetector

class Interface(tk.Tk):
    def __init__(self, screenName = None, baseName = None, className = "Tk", useTk = True, sync = False, use = None)->None:
        super().__init__(screenName, baseName, className, useTk, sync, use)
        self.title("COM31006 Assignment")


        tk.Button(self, text="embed", width=12, command=self._embed,).grid(
            row=0,column=0,padx=10, pady=10
        )
        tk.Button(self, text="verify", width=12, command=self._verify,).grid(
                    row=0,column=1,padx=10, pady=10
                )
        tk.Button(self, text="temperd", width=12, command=self._temperd,).grid(
                    row=0,column=2,padx=10, pady=10
                )
        

        self.last_meta_data=None
        self.last_watemark=None

    def _embed(self, )->None:
        carrier_img = filedialog.askopenfilename(title="Select carrier image")
        if not carrier_img:
            return
        watermark = filedialog.askopenfilename(title="Select watermark image")
        if not watermark:
            return
        
        try:
            emd = WatermarkEmbedder(carrier_image_path=carrier_img, watermark_image_path=watermark)
            output_path=emd.embed()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self._last_meta = output_path["meta_path"]
        self._last_watermark = watermark
        messagebox.showinfo("Success", f"Watermarked image saved to\n{output_path['img_path']}")
        
    def _verify(self)->None:
        if not self._last_meta:
            messagebox.showinfo(
                "Info",
                "Embed first (or manually provide metadata and watermark)",
            )
            return
        suspect = filedialog.askopenfilename(title="Select image to verify")
        if not suspect:
            return

        verifier = WatermarkVeryfier(
            img_path=suspect,
            meta_oath=self._last_meta,
            watermark_path=self._last_watermark,
        )
        auth, _, _ = verifier.verify()
        messagebox.showinfo("Result", "Authentic" if auth else "Not authentic")
    def _temperd(self)->None:
        suspect = filedialog.askopenfilename(title="Select image for tamper check")
        if not suspect:
            return
        meta = (self._last_meta or filedialog.askopenfilename(title="Select meta JSON"))
        watermark = (
            self._last_watermark
            or filedialog.askopenfilename(title="Select original watermark image")
        )
        if not meta or not watermark:
            return

        # from tamper_detector import TamperDetector

        detector = TamperDetector(
            img_path=suspect,meta_oath=meta, watermark_path=watermark
        )
        result = detector.detect()
        if result["tampered"]:
            msg = "TAMPERING DETECTED. See overlay: " + str(result["overley_tempred_path"])
        else:
            msg = "No tampering found."
        messagebox.showinfo("Tamper Detection", msg)