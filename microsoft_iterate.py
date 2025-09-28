import os
import json
import re
import sys
from pathlib import Path
from paddleocr import PaddleOCR
input_folder = sys.argv[1]

# output_folder = sys.argv[2]
# basename = Path(input_image).stem #filename without extension
# basename = basename+"_out"
# foldername = sys.argv[2]
# foldername = foldername + "/"
# Create the output folder
# os.makedirs(output_folder, exist_ok=True)

ocr = PaddleOCR(
    # Model selection (NEW parameter names in 3.x)
    text_detection_model_name="PP-OCRv5_server_det",  # Ultra-lightweight detection
    # text_recognition_model_name="en_PP-OCRv5_mobile_rec",  # Ultra-lightweight recognition
    text_recognition_model_name="PP-OCRv5_server_rec",  # Ultra-lightweight recognition
    
    # Device specification (NEW in 3.x - replaces use_gpu)
    device="cpu",  # Critical: Force CPU-only mode
    
    # CPU optimization parameters
    enable_mkldnn=True,  # Enable Intel MKL-DNN acceleration
    cpu_threads=8,       # Adjust based on your CPU cores
    mkldnn_cache_capacity=10  # MKL-DNN cache optimization
    
)
text_lines = []
for file in sorted(os.listdir(input_folder)):
    if file.endswith(".png"):
        match = re.search(r"output_frame_(\d\d\d)[\w]+_screen_([1-3]).png", file)
        
        if match:
            frame_number = match.group(1)
            screen_number = match.group(2)
            print(file, frame_number, screen_number)
            result = ocr.predict(input_folder + "/" + file)
            this_temp = 0.0
            this_conf = 0.0
            for ii,line in enumerate(result[0]['rec_texts']):
                match_temp = re.match(r'^\d+\.\d+$', line)
                if match_temp:
                    this_temp = line
                    this_conf = result[0]['rec_scores'][ii]
                    
            this_screen={
                "frame": frame_number,
                "screen": screen_number,
                "txt": result[0]['rec_texts'],
                "results_temp": this_temp,
               "results_conf": this_conf
            }


            print(this_screen)
            text_lines.append(this_screen)
            
with open("output.json", "w") as outfile:
    json.dump(text_lines, outfile)
