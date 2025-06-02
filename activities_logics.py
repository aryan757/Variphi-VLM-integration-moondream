import moondream as md
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import os
from dotenv import load_dotenv  # type: ignore

# # MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")

# # print("API KEY:", MOONDRM_API_KEY)
# model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI2YzkyNTVjMy04YTY5LTQyODgtOGMwNi1mNzljNjJlODFjYmMiLCJvcmdfaWQiOiJmeWdKb1ZsS2l6SGF4dGkwTEk4YjFjcUlkV3ZLd3BDUyIsImlhdCI6MTc0ODMyOTkzMCwidmVyIjoxfQ.nmqSfQsExP_yxZ6ezicg1qCb8mSAfQdg5qlCjCDOgug")


# MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")
# print("API KEY:", MOONDRM_API_KEY)

load_dotenv()

MOONDRM_API_KEY = os.getenv("MOONDRM_API_KEY")
print("API KEY:", MOONDRM_API_KEY)
model = md.vl(api_key=MOONDRM_API_KEY)



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
        activities = ["fire", "smoke", "phone usage", "spillage","waste"]
        prompt = (
            "Does this image contain any of the following: fire, smoke, phone usage, spillage , waste ? "
            "List only the ones that are present, separated by commas. "
            "If none are present, say 'none'."
        )
        query_response = model.query(image, prompt)
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
                result = model.detect(image, object_name)
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
                ov.save(buf, format='JPEG')
                buf.seek(0)
                filename = f"detected_{object_name}.jpg"
                ov.save(filename)
                print("filename", filename)
                print("boxes", boxes)
                results[label] = {"detected_image_url": filename, "boxes": boxes}
        return results

if __name__ == "__main__":
    print("=== Activity Detection Test ===")
    input_image = "/Users/aryan/Desktop/moondream-variphi/testing_image/imagee_4.jpg"
    service = MoondreamService()

    # Load the image from file or URL
    if input_image.startswith('http://') or input_image.startswith('https://'):
        img = service.load_image_from_url(input_image)
    else:
        img = service.load_image_from_file(input_image)

    detections = service.detect_activities(img)

    print(detections)
    print("\nDetections:")



