# import moondream as md
# from PIL import Image, ImageDraw, ImageFont
# import requests
# from io import BytesIO
# import os
# import json
# import time
# from dotenv import load_dotenv  # type: ignore
# import random

# load_dotenv()
# MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")
# # print("API KEY:", MOONDRM_API_KEY)

# model = md.vl(api_key=MOONDRM_API_KEY)



#MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")

# print("API KEY:", MOONDRM_API_KEY)
# model = md.vl(api_key=MOONDRM_API_KEY)







# ########################################################################################################################



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


model = AutoModelForCausalLM.from_pretrained(
"vikhyatk/moondream2",
revision="2025-01-09",
trust_remote_code=True, 
device_map={"": "mps"},  
)



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
        Uses model.query to ask which violations are present, then extracts a list by string matching.
        """
        activities = ["fire","smoke", "phone usage","spillage","waste"]
        prompt = (
            "Does this image contain any of the following: fire, smoke, phone usage, spillage , waste ? "
            "List only the ones that are present, separated by commas. "
            "If none are present, say 'none'."
        )
        #time.sleep(5)
        query_response = model.query(image, prompt)
        #time.sleep(1)
        response_text = query_response
        print("response_text", response_text)
        print("type of response_text", type(response_text))
        ## from the response text, extract the activities that are present , like extract only ["smoke","phone" ,"smoke","fire"] from the response text use keyword extraction

        activities = [
            "fire",
            "smoke",
            "phone usage",
            "spillage",
            "waste",
        ]

        present_activities = []
        for activity in activities:
            if activity in query_response["answer"]:
                present_activities.append(activity)
        if not present_activities and ("none" in response_text or "no" in response_text):
            return []
        return present_activities


    def detect_activities(self, image: Image.Image):
        # List of activities to detect with appropriate prompts/object names
        activities = [
            ("fire", "fire"),
            ("smoke", "smoke"),
            ("phone usage", "phone"),
            ("spillage", "spillage"),
            ("waste", "waste"),
        ]

        w, h = image.size
        print(w, h)
        # Step 1: Query which activities are present
        present_activities = self.get_present_activities(image)
        print("Present activities:", present_activities)
        results = {}
        for label, object_name in activities:
            if label in present_activities:
                print("Detecting:", label)
                #time.sleep(5)
                result = model.detect(image, object_name)
                #time.sleep(1)
                print("result", result)
                boxes = result["objects"]
                print("boxes", boxes)
                ov = image.copy()
                d = ImageDraw.Draw(ov)
                for b in boxes:
                    d.rectangle([
                        int(b['x_min'] * w),
                        int(b['y_min'] * h),
                        int(b['x_max'] * w),
                        int(b['y_max'] * h)
                    ], outline='red', width=2)
                buf = BytesIO()
                ov.save(buf, format='JPEG'
                        )
                buf.seek(0)
                filename = f"detected_{object_name}_{random.randint(1, 1000)}.jpg"
                #save the each detected image  in the output_inference folder
                output_inference_folder = "output_inference_4"
                if not os.path.exists(output_inference_folder):
                    os.makedirs(output_inference_folder)
                ov.save(os.path.join(output_inference_folder, filename))
                #save the detected image in the output_inference folder
                #save the detected image in the output_inference folder
                #ov.save(filename)
                print("filename", filename)
                print("boxes", boxes)
                results[label] = {"detected_image_url": filename, "boxes": boxes}
        return results

# if __name__ == "__main__":
#     print("=== Activity Detection Test ===")
#     input_image = "/Users/aryan/Desktop/moondream-variphi/testing_image/imagee_4.jpg"
#     service = MoondreamService()

#     # Load the image from file or URL
#     if input_image.startswith('http://') or input_image.startswith('https://'):
#         img = service.load_image_from_url(input_image)
#     else:
#         img = service.load_image_from_file(input_image)

#     detections = service.detect_activities(img)

#     print(detections)
#     print("\nDetections:")

##write a logoc inthe main function to iterate every image in the Datasets/Fire & Smoke folder to detect , inside this folder
## there are 2 subfolders in this folder , Fire and Smoke , iterate through each of them and detect the activities
## save the results in a json file , the json file should have the image name and the activities detected in the image
## the json file should be saved in the same folder as the images
## the json file should be named as activities_detection_results.json
## the json file should have the image name and the activities detected in the image
## the json file should be saved in the same folder as the images
## the json file should be named as activities_detection_results.json
#leave the video for now

if __name__ == "__main__":
    service = MoondreamService()
    #for folder in ["fire", "smoke"]:
    for folder in ["fire"]:
        print("folder", folder)
       # folder_path = os.path.join("Datasets","CCTV",folder)
        folder_path = os.path.join("Datasets",folder)
        
        #print("folder_path", folder_path)
        results = {}
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_file)
                print(f"Detecting activities in: {image_path}")
                image = service.load_image_from_file(image_path)
                detections = service.detect_activities(image)
                results[image_file] = detections
                print(f"Detections for {image_file}:", detections)
        
        # Save results to JSON file in the respective folder
        # output_file = os.path.join(folder_path, "activities_detection_results.json")
        # with open(output_file, 'w') as f:
        #     json.dump(results, f, indent=4)
        # print(f"Results saved to {output_file}")
















