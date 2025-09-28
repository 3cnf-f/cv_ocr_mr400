import os
import json
import re
import sys
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
input_folder = sys.argv[1]

# output_folder = sys.argv[2]
# basename = Path(input_image).stem #filename without extension
# basename = basename+"_out"
# foldername = sys.argv[2]
# foldername = foldername + "/"
# Create the output folder
# os.makedirs(output_folder, exist_ok=True)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

text_lines = []
for file in sorted(os.listdir(input_folder)):
    if file.endswith(".png"):
        match = re.search(r"output_frame_(\d\d\d)[\w]+_screen_([1-3]).png", file)
        
        if match:
            frame_number = match.group(1)
            screen_number = match.group(2)
            print(file, frame_number, screen_number)
            image = Image.open(input_folder + "/" + file).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            this_temp = 0.0
            this_conf = 0.0
            print("\n\n",frame_number, screen_number,"\n", generated_text)
#             for ii,line in enumerate(result[0]['rec_texts']):
#                 match_temp = re.match(r'^\d+\.\d+$', line)
#                 if match_temp:
#                     this_temp = line
#                     this_conf = result[0]['rec_scores'][ii]
#
#             this_screen={
#                 "frame": frame_number,
#                 "screen": screen_number,
#                 "txt": result[0]['rec_texts'],
#                 "results_temp": this_temp,
#                "results_conf": this_conf
#             }
#
#
#             print(this_screen)
#             text_lines.append(this_screen)
#
# with open("output.json", "w") as outfile:
#     json.dump(text_lines, outfile)
