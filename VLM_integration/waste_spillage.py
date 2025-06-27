
import moondream as md
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import os
import json
import time
from dotenv import load_dotenv  # type: ignore
import random
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import os
import json
import time
from dotenv import load_dotenv  # type: ignore
import random
import cv2



##########################################---------API MODELs---------##########################################

load_dotenv()

# MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
MOONDREAM_API_SECOND_KEY = os.getenv("MOONDREAM_API_SECOND_KEY")

# print("API KEY 1:", MOONDREAM_API_KEY)
# print("API KEY 2:", MOONDREAM_API_SECOND_KEY)
# model = md.vl(api_key=MOONDREAM_API_KEY)
# model_2 = md.vl(api_key=MOONDREAM_API_SECOND_KEY)

model = md.vl(api_key=MOONDREAM_API_SECOND_KEY)


##########################################---------LOCAL MODEL HUGGING FACE---------##########################################



# Check for available devices
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU")




# model = AutoModelForCausalLM.from_pretrained(
# "vikhyatk/moondream2",
# revision="2025-01-09",
# trust_remote_code=True, 
# device_map={"": "mps"},  
# )



##########################################---------LOCAL QUANTIZED MODEL---------##########################################

# # Check for available devices
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print("Using MPS device")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using CUDA device")
# else:
#     device = torch.device("cpu")
#     print("Using CPU")



# model = AutoModelForCausalLM.from_pretrained(
#     "moondream/moondream-2b-2025-04-14-4bit",
#     trust_remote_code=True,
#     device_map={"": "mps"}
# )

# # Optional, but recommended when running inference on a large number of
# # images since it has upfront compilation cost but significantly speeds
# # up inference:
# model.model.compile()

  
########################################################################################################################



class MoondreamService:
    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    @staticmethod
    def load_image_from_file(file_path: str) -> Image.Image:
        return Image.open(file_path)

    def get_present_activities(self, image: Image.Image):
        """
        Ask the model if aggressive behaviour, violence, or physical fights are present.
        If yes, return the one-liner with location info.
        If no, return None.
        """

        # prompt = (
        #     "Does this image contain any of the following: waste, spillage ?"
        #     "Respond only the ones that are present, separated by commas."
        #     "If none are present, say 'none'."
        # )

        ## write the same way the way you have written in the phone_usage.py file , here write the prompt for waste and spillage
        prompt = (
            "Does this image contain any of the following: waste or spillage ?"
            "If yes, respond with 'Yes' followed by the one-liner clearly describing the entity involved and their approximate position in the frame. "
            "Use location terms like 'on the top left', 'center', 'extreme right', etc. "
            "If no such activity is visible, respond with 'None'."
        )

        query_response = model.query(image, prompt)
        print("query_response========>", query_response)
        response_text = query_response["answer"] if isinstance(query_response, dict) else query_response
        if "Yes" or "yes" and "waste" or "spillage" or "Waste" or "Spillage" in response_text.lower():
            return "waste" + " " + "spillage" + " " + "Waste" + " " + "Spillage"
        print("Query Response:", response_text)
        if any(word in response_text.lower() for word in ["none", "no","No","None"]):
            return None
        return response_text.strip()


    def point_activities(self, image: Image.Image, get_present_activities: str):
        w, h = image.size
        print("w,h", w, h)
        result = model.point(image, get_present_activities)
        print("result", result)
        points = result["points"]
        print("points", points)
        ov = image.copy()
        if ov.mode == 'RGBA':
            ov = ov.convert('RGB')
        draw = ImageDraw.Draw(ov)
        for pt in points:
            r = 10
            draw.ellipse([
                int(pt['x'] * w) - r, int(pt['y'] * h) - r,
                int(pt['x'] * w) + r, int(pt['y'] * h) + r
            ], fill='blue')
        buf = BytesIO()
        ov.save(buf, format='JPEG')
        buf.seek(0)
        filename = f"detected_{random.randint(1, 1000)}.jpg"
        if not os.path.exists("spillage_point_output"):
            os.makedirs("spillage_point_output", exist_ok=True)
        ov.save(os.path.join("spillage_point_output", filename))
        print("filename", filename)
        print("points", points)
        return filename


    def detect_activities(self, image: Image.Image, get_present_activities: str):
        w, h = image.size
        print("w,h", w, h)
        result = model.detect(image, get_present_activities)
        print("result", result)
        boxes = result["objects"]
        print("boxes", boxes)
        ov = image.copy()
        if ov.mode == 'RGBA':
            ov = ov.convert('RGB')
        d = ImageDraw.Draw(ov)
        for b in boxes:
            d.rectangle([
                int(b['x_min'] * w),
                int(b['y_min'] * h),
                int(b['x_max'] * w),
                int(b['y_max'] * h)
            ], outline='red', width=2)
        buf = BytesIO()
        ov.save(buf, format='JPEG')
        buf.seek(0)
        filename = f"detected_{random.randint(1, 1000)}.jpg"
        if not os.path.exists("spillage_detect_output"):
            os.makedirs("spillage_detect_output", exist_ok=True)
        ov.save(os.path.join("spillage_detect_output", filename))
        print("filename", filename)
        print("boxes", boxes)
        return filename


def process_images(image_folder, output_dir="output_images"):
    """
    Process all images in the specified folder.
    Args:
        image_folder (str): Path to the folder containing images.
        output_dir (str): Directory to save output results.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    if not image_files:
        print(f"No image files found in '{image_folder}'")
        return
    print(f"Found {len(image_files)} images to process in '{image_folder}'")
    service = MoondreamService()
    for idx, image_path in enumerate(image_files, 1):
        try:
            image = Image.open(image_path).convert("RGB")
            activities = service.get_present_activities(image)
            if activities:

                point_file = service.point_activities(image, activities)
                box_file = service.detect_activities(image, activities)

                print(f"[{idx}/{len(image_files)}] Processed: {os.path.basename(image_path)} | Activities: {activities}")
                # Optionally, copy/move result files to output_dir if needed

            else:

                print(f"[{idx}/{len(image_files)}] Processed: {os.path.basename(image_path)} | No activities detected")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def process_videos(video_folder, output_dir="output_videos", frame_interval=100):
    """
    Process all videos in the specified folder, extracting every Nth frame.
    Args:
        video_folder (str): Path to the folder containing videos.
        output_dir (str): Directory to save output results.
        frame_interval (int): Extract and process every Nth frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = [
        os.path.join(video_folder, f)
        for f in os.listdir(video_folder)
        if os.path.splitext(f.lower())[1] in video_extensions
    ]
    if not video_files:
        print(f"No video files found in '{video_folder}'")
        return
    print(f"Found {len(video_files)} videos to process in '{video_folder}'")
    service = MoondreamService()
    for vid_idx, video_path in enumerate(video_files, 1):
        print(f"\nProcessing video [{vid_idx}/{len(video_files)}]: {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                activities = service.get_present_activities(pil_image)
                if activities:
                    point_file = service.point_activities(pil_image, activities)
                    box_file = service.detect_activities(pil_image, activities)
                    print(f"  Frame {frame_count}: Activities detected - {activities}")
                    saved_frame_count += 1
                else:
                    print(f"  Frame {frame_count}: No activities detected")
            frame_count += 1
        cap.release()
        print(f"Processed {saved_frame_count} frames from {os.path.basename(video_path)}")
    print(f"All videos processed. Results saved in '{output_dir}'.")

def main():
    """
    Main function to process images or videos.
    Edit the function calls below to specify your folder locations.
    """
    # Example usage:
    # To process images in a folder:
    # process_images("path/to/your/image_folder", output_dir="your_output_dir")
    #
    # To process videos in a folder:
    # process_videos("fire", output_dir="output_inference_7", frame_interval=100)
    # print("==================done- videos===================")

    # --- EDIT BELOW AS NEEDED ---
    # Uncomment and set your folder paths

    # Example: process images
    process_images("waste_spillage_data", output_dir="output_inference_7")
    print("==================done- images===================")

    # Example: process videos
    # process_videos("video-dataset", output_dir="output-dir", frame_interval=100)

    # For demonstration, you can uncomment one of the following lines:
    # process_images("image-dataset")
    # process_videos("video-dataset")

    print("Edit the main() function to call process_images() or process_videos() with your folder paths.")

if __name__ == "__main__":
    main()

