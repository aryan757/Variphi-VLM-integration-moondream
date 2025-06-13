# import moondream as md
# from PIL import Image, ImageDraw, ImageFont
# import requests
# from io import BytesIO
# import os
# import json
# import time
# from dotenv import load_dotenv  # type: ignore
# import random
# # MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")

# # print("API KEY:", MOONDRM_API_KEY)
# model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI2YzkyNTVjMy04YTY5LTQyODgtOGMwNi1mNzljNjJlODFjYmMiLCJvcmdfaWQiOiJmeWdKb1ZsS2l6SGF4dGkwTEk4YjFjcUlkV3ZLd3BDUyIsImlhdCI6MTc0ODMyOTkzMCwidmVyIjoxfQ.nmqSfQsExP_yxZ6ezicg1qCb8mSAfQdg5qlCjCDOgug")

# load_dotenv()
# MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")
# # print("API KEY:", MOONDRM_API_KEY)

# model = md.vl(api_key=MOONDRM_API_KEY)

########################################################################################################################




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
        #activities = ["fire","smoke", "phone usage","spillage","waste"]
        prompt = (
            "Is there any person working in a dangerous condition, such as near an edge or at a height, in this image? "
            "If yes, mention the approximate location (e.g., top-left, center, bottom-right). "
            "If not, reply 'No dangerous activity detected'."
        )

        time.sleep(5)
        query_response = model.query(image, prompt)
        time.sleep(1)
        response_text = query_response
        print("response_text", response_text)
        print("type of response_text", type(response_text))
        ## from the response text, extract the activities that are present , like extract only ["smoke","phone" ,"smoke","fire"] from the response text use keyword extraction

        # activities = [
        #     "fire",
        #     "smoke",
        #     "phone usage",
        #     "spillage",
        #     "waste",
        # ]

        # present_activities = []
        # for activity in activities:
        #     if activity in query_response["answer"]:
        #         present_activities.append(activity)
        # if not present_activities and ("none" in response_text or "no" in response_text):
        #     return []
        # return present_activities



if __name__ == "__main__":
    print("=== Activity Detection Test ===")
    testing_folder = "/Users/aryan/Desktop/moondream-variphi/testing_image"
    service = MoondreamService()

    for filename in os.listdir(testing_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_image = os.path.join(testing_folder, filename)
            print(f"Processing: {input_image}")
            # Load the image from file or URL
            if input_image.startswith('http://') or input_image.startswith('https://'):
                img = service.load_image_from_url(input_image)
            else:
                img = service.load_image_from_file(input_image)

            query_response = service.get_present_activities(img)
            print(query_response)



