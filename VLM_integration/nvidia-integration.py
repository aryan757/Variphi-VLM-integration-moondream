import json
import io
import base64
import time
import os
import random
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# NVIDIA VLM API URLs
neva_api_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
kosmos2_api_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2"
fuyu8b_api_url = "https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b"
paligemma_api_url = "https://ai.api.nvidia.com/v1/vlm/google/paligemma"
phi3_api_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"

# API Key
api_key = "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"

class VLM:
    def __init__(self, url, api_key):
        """ Provide NIM API URL and an API key"""
        self.api_key = api_key
        self.url = url
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}

    def _encode_image(self, image):
        """ Resize image, encode as jpeg to shrink size then convert to b64 for upload """
        if isinstance(image, str):  # file path
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):  # pil image
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):  # cv2 / np array image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        else:
            print(f"Unsupported image input: {type(image)}")
            return None

        image = image.resize((336, 336))
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image = buf.getvalue()
        image_b64 = base64.b64encode(image).decode()
        assert len(image_b64) < 180_000, "Image too large to upload."
        return image_b64

    def __call__(self, prompt, image, max_tokens=512):
        """ Call VLM object with the prompt and path to image """
        image_b64 = self._encode_image(image)

        # For simplicity, the image will be appended to the end of the prompt.
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} Here is the image: <img src="data:image/jpeg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.20,
            "top_p": 0.70,
            "stream": False
        }

        response = requests.post(self.url, headers=self.headers, json=payload)
        response = response.json()
        if "choices" in response and len(response["choices"]) > 0:
            reply = response["choices"][0]["message"]["content"]
        else:
            reply = "Error: Could not get a response from the VLM."
            print("Full response from VLM API:")
            print(json.dumps(response, indent=4, sort_keys=True))

        return reply, response


class NvidiaVLMService:
    def __init__(self, model_name="neva"):
        """
        Initialize the NVIDIA VLM service with specified model.
        Available models: neva, phi3, fuyu8b, kosmos2, paligemma
        """
        model_urls = {
            "neva": neva_api_url,
            "phi3": phi3_api_url,
            "fuyu8b": fuyu8b_api_url,
            "kosmos2": kosmos2_api_url,
            "paligemma": paligemma_api_url
        }
        
        if model_name not in model_urls:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(model_urls.keys())}")
        
        self.model = VLM(model_urls[model_name], api_key)
        self.model_name = model_name
        print(f"Initialized NVIDIA VLM Service with {model_name} model")

    @staticmethod
    def load_image_from_url(url: str) -> Image.Image:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    @staticmethod
    def load_image_from_file(file_path: str) -> Image.Image:
        return Image.open(file_path)

    def get_present_activities(self, image: Image.Image, activity_type="general"):
        """
        Ask the model about specific activities present in the image.
        """
        prompts = {
            "general": (
                "Analyze this image and identify any suspicious or concerning activities. "
                "Look for: aggressive behavior, violence, physical fights, property tampering, "
                "vandalism, theft, shoplifting, phone usage during work, waste spillage, "
                "fire, smoke, or safety violations. "
                "If any activities are detected, respond with 'Yes' followed by a detailed description "
                "including the person's approximate position in the frame (e.g., 'left side', 'center', 'right'). "
                "If no concerning activities are visible, respond with 'None'."
            ),
            "violence": (
                "Does this image show any form of aggressive behavior, violence, or physical fights? "
                "If yes, respond with 'Yes' followed by a one-liner clearly describing the person involved "
                "and their approximate position in the frame. Use location terms like 'the person on the top left', "
                "'center', 'extreme right', etc. If no such activity is visible, respond with 'None'."
            ),
            "property_tampering": (
                "Does this image show any form of property tampering or vandalism? "
                "If yes, respond with 'Yes' followed by a one-liner clearly describing the person involved "
                "and their approximate position in the frame. Use location terms like 'the person on the top left', "
                "'center', 'extreme right', etc. If no such activity is visible, respond with 'None'."
            ),
            "theft": (
                "Does this image show any signs of theft or shoplifting? "
                "If yes, respond with 'Yes' followed by a one-liner clearly describing the person involved "
                "and their approximate position in the frame. If no such activity is visible, respond with 'None'."
            ),
            "phone_usage": (
                "Does this image show someone using a phone during work hours in a workplace setting? "
                "If yes, respond with 'Yes' followed by a description of the person and their location in the frame. "
                "If no phone usage is visible, respond with 'None'."
            ),
            "waste_spillage": (
                "Does this image contain any of the following: waste, spillage, garbage, or spilled materials? "
                "Respond only with the ones that are present, separated by commas. "
                "If none are present, say 'None'."
            ),
            "fire_smoke": (
                "Does this image show any signs of fire, smoke, flames, or fire-related hazards? "
                "If yes, describe what you see and its location in the image. "
                "If no fire or smoke is visible, respond with 'None'."
            )
        }

        prompt = prompts.get(activity_type, prompts["general"])
        
        try:
            response, _ = self.model(prompt, image, max_tokens=256)
            print(f"Query Response ({self.model_name}): {response}")
            
            if any(word in response.lower() for word in ["none", "no such", "not visible", "no signs"]):
                return None
            return response.strip()
        except Exception as e:
            print(f"Error in get_present_activities: {e}")
            return None

    def analyze_with_bounding_box(self, image: Image.Image, detected_activity: str):
        """
        Analyze the image and try to provide bounding box information for detected activities.
        """
        prompt = (
            f"In this image, you detected: '{detected_activity}'. "
            "Can you provide more specific location details about where this activity is occurring? "
            "Describe the approximate coordinates or percentage position from the top-left corner "
            "(e.g., 'approximately 30% from left, 40% from top'). "
            "Also estimate the size of the area where this activity is happening."
        )
        
        try:
            response, _ = self.model(prompt, image, max_tokens=512)
            return response.strip()
        except Exception as e:
            print(f"Error in analyze_with_bounding_box: {e}")
            return "Could not determine specific location"

    def create_annotated_image(self, image: Image.Image, activity_description: str, 
                             output_folder: str = "nvidia_vlm_output"):
        """
        Create an annotated image with detected activities marked.
        Since we don't have exact coordinates, we'll add text overlay.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Create a copy of the image
            annotated_image = image.copy()
            if annotated_image.mode == 'RGBA':
                annotated_image = annotated_image.convert('RGB')
            
            # Add text overlay
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Add background rectangle for text
            text_bbox = draw.textbbox((10, 10), f"Detected: {activity_description[:100]}...", font=font)
            draw.rectangle([text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5], 
                         fill='yellow', outline='red', width=2)
            
            # Add text
            draw.text((10, 10), f"Detected: {activity_description[:100]}...", 
                     fill='red', font=font)
            
            # Save the annotated image
            filename = f"nvidia_detected_{random.randint(1, 10000)}.jpg"
            filepath = os.path.join(output_folder, filename)
            annotated_image.save(filepath)
            
            print(f"Annotated image saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error creating annotated image: {e}")
            return None


def process_images(image_folder, activity_type="general", model_name="neva", output_dir="nvidia_vlm_output"):
    """
    Process all images in the specified folder using NVIDIA VLM API.
    Args:
        image_folder (str): Path to the folder containing images.
        activity_type (str): Type of activity to detect.
        model_name (str): NVIDIA model to use.
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
    service = NvidiaVLMService(model_name=model_name)
    
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            image = Image.open(image_path).convert("RGB")
            
            # Detect activities
            activities = service.get_present_activities(image, activity_type)
            
            if activities:
                # Get more detailed analysis
                detailed_analysis = service.analyze_with_bounding_box(image, activities)
                
                # Create annotated image
                annotated_filename = service.create_annotated_image(image, activities, output_dir)
                
                result = {
                    "image": os.path.basename(image_path),
                    "activities_detected": activities,
                    "detailed_analysis": detailed_analysis,
                    "annotated_image": annotated_filename,
                    "model_used": model_name
                }
                results.append(result)
                
                print(f"✓ Activities detected: {activities}")
                print(f"✓ Detailed analysis: {detailed_analysis[:100]}...")
                
            else:
                result = {
                    "image": os.path.basename(image_path),
                    "activities_detected": "None",
                    "detailed_analysis": "No concerning activities detected",
                    "annotated_image": None,
                    "model_used": model_name
                }
                results.append(result)
                print(f"✓ No activities detected")
                
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")
            result = {
                "image": os.path.basename(image_path),
                "activities_detected": "Error",
                "error": str(e),
                "model_used": model_name
            }
            results.append(result)
    
    # Save results to JSON
    results_file = os.path.join(output_dir, f"nvidia_vlm_results_{model_name}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processing complete! Results saved to: {results_file}")
    return results


def process_videos(video_folder, activity_type="general", model_name="neva", 
                  output_dir="nvidia_vlm_video_output", frame_interval=100):
    """
    Process all videos in the specified folder, extracting every Nth frame.
    Args:
        video_folder (str): Path to the folder containing videos.
        activity_type (str): Type of activity to detect.
        model_name (str): NVIDIA model to use.
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
    service = NvidiaVLMService(model_name=model_name)
    
    all_results = []
    
    for vid_idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing video [{vid_idx}/{len(video_files)}]: {os.path.basename(video_path)}")
        print(f"{'='*50}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detection_count = 0
        video_results = []
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps} FPS, {duration:.2f}s duration")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                try:
                    # Convert frame to PIL Image
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    # Detect activities
                    activities = service.get_present_activities(pil_image, activity_type)
                    
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    if activities:
                        # Get detailed analysis
                        detailed_analysis = service.analyze_with_bounding_box(pil_image, activities)
                        
                        # Create annotated image for this frame
                        frame_output_dir = os.path.join(output_dir, f"video_{vid_idx}_frames")
                        annotated_filename = service.create_annotated_image(
                            pil_image, activities, frame_output_dir)
                        
                        result = {
                            "frame": frame_count,
                            "timestamp": f"{timestamp:.2f}s",
                            "activities_detected": activities,
                            "detailed_analysis": detailed_analysis,
                            "annotated_frame": annotated_filename
                        }
                        video_results.append(result)
                        detection_count += 1
                        
                        print(f"  🔍 Frame {frame_count} ({timestamp:.2f}s): {activities[:80]}...")
                        
                    else:
                        print(f"  ✓ Frame {frame_count} ({timestamp:.2f}s): No activities detected")
                        
                except Exception as e:
                    print(f"  ✗ Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Save video results
        video_result_summary = {
            "video_file": os.path.basename(video_path),
            "total_frames": total_frames,
            "frames_processed": frame_count // frame_interval,
            "detections_found": detection_count,
            "fps": fps,
            "duration": f"{duration:.2f}s",
            "model_used": model_name,
            "activity_type": activity_type,
            "frame_results": video_results
        }
        all_results.append(video_result_summary)
        
        print(f"✓ Video processed: {detection_count} detections in {len(video_results)} analyzed frames")
    
    # Save all results to JSON
    results_file = os.path.join(output_dir, f"nvidia_vlm_video_results_{model_name}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 All videos processed! Results saved to: {results_file}")
    return all_results

 
def main():
    """
    Main function to process images or videos using NVIDIA VLM API.
    Edit the function calls below to specify your folder locations and preferences.
    """
    print("🚀 NVIDIA VLM Integration for Activity Detection")
    print("Available models: neva, phi3, fuyu8b, kosmos2, paligemma")
    print("Available activity types: general, violence, property_tampering, theft, phone_usage, waste_spillage, fire_smoke")
    print("="*80)
    
    # --- CONFIGURATION ---
    # Edit these parameters as needed
    
    # For processing images:
    process_images(
        image_folder="fire",
        activity_type="fire_smoke",  # or specific type
        model_name="neva",        # choose your model
        output_dir="nvidia_output"
    )
    
    # For processing videos:
    # process_videos(
    #     video_folder="path/to/your/videos",
    #     activity_type="general",
    #     model_name="neva",
    #     output_dir="nvidia_video_output",
    #     frame_interval=100  # process every 100th frame
    # )
    
    # Example usage - uncomment and modify as needed:
    
    # Process fire detection in images
    # process_images("fire", activity_type="fire_smoke", model_name="neva", output_dir="fire_detection_output")
    
    # Process theft detection in videos
    # process_videos("CCTV_real_data", activity_type="theft", model_name="phi3", 
    #               output_dir="theft_detection_output", frame_interval=50)
    
    print("Edit the main() function to process your images or videos!")
    print("Example:")
    print("process_images('your_image_folder', activity_type='violence', model_name='neva')")
    print("process_videos('your_video_folder', activity_type='theft', model_name='phi3')")


if __name__ == "__main__":
    main()
