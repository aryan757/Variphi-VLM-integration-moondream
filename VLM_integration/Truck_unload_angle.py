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

MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY")
MOONDREAM_API_SECOND_KEY = os.getenv("MOONDREAM_API_SECOND_KEY")

print("API KEY 1:", MOONDREAM_API_KEY)
print("API KEY 2:", MOONDREAM_API_SECOND_KEY)
model = md.vl(api_key=MOONDREAM_API_KEY)
model_2 = md.vl(api_key=MOONDREAM_API_SECOND_KEY)


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
        Ask the model to detect the truck unload angle.
        If a truck is unloading, return the angle measurement.
        If no truck unloading is detected, return None.
        """

        # prompt = (
        #     "Look at this image and determine if there is a truck or dump truck that is unloading material. "
        #     "If you see a truck with its bed/container tilted up for unloading, please measure and provide only the angle of the truck bed relative to horizontal ground. "
        #     "Respond with only the numerical angle value followed by 'degrees' (e.g., '45 degrees'). "
        #     "If no truck unloading is visible or the truck bed is not tilted, respond with 'none'."
        # )

        prompt = (
    "Look at this image of a truck. Focus on the angle between the tilted container (the truck bed) and the horizontal ground. "
    "You can use the rear wheels and the base of the truck as reference to estimate the angle. "
    "Provide only the estimated angle of the tilted container in degrees. "
    "Respond with just the number followed by 'degrees' (e.g., '32 degrees'). "
    "If there is no tilt, respond with 'none'."
)


        query_response = model.query(image, prompt)
        response_text = query_response["answer"] if isinstance(query_response, dict) else query_response
        print("Query Response:", response_text)
        if any(word in response_text.lower() for word in ["none", "no truck", "not"]):
            return None
        return response_text.strip()


    # COMMENTED OUT - Detection methods as requested
    # def point_activities(self, image: Image.Image, get_present_activities: str):
    #     w, h = image.size
    #     print("w,h", w, h)
    #     result = model.point(image, get_present_activities)
    #     print("result", result)
    #     points = result["points"]
    #     print("points", points)
    #     ov = image.copy()
    #     if ov.mode == 'RGBA':
    #         ov = ov.convert('RGB')
    #     draw = ImageDraw.Draw(ov)
    #     for pt in points:
    #         r = 10
    #         draw.ellipse([
    #             int(pt['x'] * w) - r, int(pt['y'] * h) - r,
    #             int(pt['x'] * w) + r, int(pt['y'] * h) + r
    #         ], fill='blue')
    #     buf = BytesIO()
    #     ov.save(buf, format='JPEG')
    #     buf.seek(0)
    #     filename = f"detected_{random.randint(1, 1000)}.jpg"
    #     if not os.path.exists("truck_angle_point_output"):
    #         os.makedirs("truck_angle_point_output", exist_ok=True)
    #     ov.save(os.path.join("truck_angle_point_output", filename))
    #     print("filename", filename)
    #     print("points", points)
    #     return filename


    # def detect_activities(self, image: Image.Image, get_present_activities: str):
    #     w, h = image.size
    #     print("w,h", w, h)
    #     result = model.detect(image, get_present_activities)
    #     print("result", result)
    #     boxes = result["objects"]
    #     print("boxes", boxes)
    #     ov = image.copy()
    #     if ov.mode == 'RGBA':
    #         ov = ov.convert('RGB')
    #     d = ImageDraw.Draw(ov)
    #     for b in boxes:
    #         d.rectangle([
    #             int(b['x_min'] * w),
    #             int(b['y_min'] * h),
    #             int(b['x_max'] * w),
    #             int(b['y_max'] * h)
    #         ], outline='red', width=2)
    #     buf = BytesIO()
    #     ov.save(buf, format='JPEG')
    #     buf.seek(0)
    #     filename = f"detected_{random.randint(1, 1000)}.jpg"
    #     if not os.path.exists("truck_angle_detect_output"):
    #         os.makedirs("truck_angle_detect_output", exist_ok=True)
    #     ov.save(os.path.join("truck_angle_detect_output", filename))
    #     print("filename", filename)
    #     print("boxes", boxes)
    #     return filename


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
            angle_result = service.get_present_activities(image)
            if angle_result:
                print(f"[{idx}/{len(image_files)}] Processed: {os.path.basename(image_path)} | Truck Unload Angle: {angle_result}")
                # Note: Detection methods are commented out as requested
                # point_file = service.point_activities(image, angle_result)
                # box_file = service.detect_activities(image, angle_result)
            else:
                print(f"[{idx}/{len(image_files)}] Processed: {os.path.basename(image_path)} | No truck unloading detected")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def process_videos(video_folder, output_dir="output_videos", frame_interval=350):
    """
    Process all videos in the specified folder, extracting every 350th frame (as requested).
    Args:
        video_folder (str): Path to the folder containing videos.
        output_dir (str): Directory to save output results.
        frame_interval (int): Extract and process every Nth frame (default: 350).
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
                angle_result = service.get_present_activities(pil_image)
                if angle_result:
                    print(f"  Frame {frame_count}: Truck Unload Angle detected - {angle_result}")
                    saved_frame_count += 1
                    # Note: Detection methods are commented out as requested
                    # point_file = service.point_activities(pil_image, angle_result)
                    # box_file = service.detect_activities(pil_image, angle_result)
                else:
                    print(f"  Frame {frame_count}: No truck unloading detected")
            frame_count += 1
        cap.release()
        print(f"Processed {saved_frame_count} frames with truck unloading from {os.path.basename(video_path)}")
    print(f"All videos processed. Results saved in '{output_dir}'.")

def main():
    """
    Main function to process images or videos for truck unload angle detection.
    Edit the function calls below to specify your folder locations.
    """
    # Example usage:
    # To process images in a folder:
    # process_images("path/to/your/image_folder", output_dir="your_output_dir")
    #
    # To process videos in a folder (with 350 frame interval):
    process_videos("truck_unloading_video_data", output_dir="truck_angle_output", frame_interval=150)
    print("==================done- videos===================")

    # --- EDIT BELOW AS NEEDED ---
    # Uncomment and set your folder paths

    # Example: process images
    # process_images("truck_unloading_data", output_dir="truck_angle_output")
    # print("==================done- images===================")

    # Example: process videos with 350 frame interval
    # process_videos("truck_videos", output_dir="truck_angle_output", frame_interval=350)

    # For demonstration, you can uncomment one of the following lines:
    # process_images("image-dataset")
    # process_videos("video-dataset", frame_interval=350)

    #print("Edit the main() function to call process_images() or process_videos() with your folder paths.")
    #print("Frame interval is set to 350 as requested for video processing.")

if __name__ == "__main__":
    main()
