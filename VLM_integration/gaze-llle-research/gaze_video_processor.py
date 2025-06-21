import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import cv2
import os
from retinaface import RetinaFace
import time

def setup_model():
    """Setup the gaze detection model"""
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    print(f"Using device: {device}")
    
    # load Gaze-LLE model
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    model.eval()
    model.to(device)
    
    return model, transform, device

def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None):
    """Visualize predicted gaze heatmap for each person and gaze in/out of frame score"""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))
    return overlay_image

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    """Combined visualization with maximal gaze points for each person"""
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=int(min(width, height) * 0.01))

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill=color, font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], fill=color, width=int(0.005*min(width, height)))

    return overlay_image

def process_frame(frame, model, transform, device, frame_number):
    """Process a single frame for gaze detection"""
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    width, height = image.size

    # detect faces
    resp = RetinaFace.detect_faces(np.array(image))
    if not resp:
        print(f"Frame {frame_number}: No faces detected")
        return image, None
    
    print(f"Frame {frame_number}: Detected {len(resp)} faces")
    bboxes = [resp[key]['facial_area'] for key in resp.keys()]

    # prepare gazelle input
    img_tensor = transform(image).unsqueeze(0).to(device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

    input_data = {
        "images": img_tensor, # [num_images, 3, 448, 448]
        "bboxes": norm_bboxes # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
    }

    with torch.no_grad():
        output = model(input_data)

    # Create combined visualization
    result_image = visualize_all(
        image, 
        output['heatmap'][0], 
        norm_bboxes[0], 
        output['inout'][0] if output['inout'] is not None else None, 
        inout_thresh=0.5
    )
    
    return result_image, output


def process_single_image(image_path, model, transform, device, output_dir):
    """Process a single image for gaze detection (only save combined gaze detection result)"""
    try:
        # Read image using OpenCV
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return False

        print(f"Processing image {os.path.basename(image_path)}...")

        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        width, height = image.size

        # detect faces
        resp = RetinaFace.detect_faces(np.array(image))
        if not resp:
            print(f"No faces detected in {os.path.basename(image_path)}")
            # Save original image if no faces detected
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{image_name}_no_faces.png")
            image.save(output_path)
            return False

        print(f"Detected {len(resp)} faces in {os.path.basename(image_path)}")
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]

        # prepare gazelle input
        img_tensor = transform(image).unsqueeze(0).to(device)
        norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

        input_data = {
            "images": img_tensor,  # [num_images, 3, 448, 448]
            "bboxes": norm_bboxes  # [[img1_bbox1, img1_bbox2...], [img2_bbox1, img2_bbox2]...]
        }

        with torch.no_grad():
            output = model(input_data)

        # Get image name without extension for output naming
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Only create and save combined gaze detection visualization
        combined_result = visualize_all(
            image,
            output['heatmap'][0],
            norm_bboxes[0],
            output['inout'][0] if output['inout'] is not None else None,
            inout_thresh=0.5
        )
        combined_path = os.path.join(output_dir, f"{image_name}_combined_result.png")
        combined_result.save(combined_path)
        print(f"Saved combined result to {combined_path}")

        return True

    except Exception as e:
        print(f"Error processing image {os.path.basename(image_path)}: {e}")
        return False

def process_images(image_folder, output_dir="heatmaps"):
    """
    Process all images in the specified folder.
    Args:
        image_folder (str): Path to the folder containing images.
        output_dir (str): Directory to save output results.
    """
    print("Setting up model...")
    model, transform, device = setup_model()
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
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
    for idx, image_path in enumerate(image_files, 1):
        start_time = time.time()
        success = process_single_image(image_path, model, transform, device, output_dir)
        end_time = time.time()
        print(f"[{idx}/{len(image_files)}] Processed: {os.path.basename(image_path)} | Success: {success} | Time: {end_time - start_time:.2f}s")

    print(f"All images processed. Results saved in '{output_dir}'.")

def process_videos(video_folder, output_dir="video_heatmaps", frame_interval=100):
    """
    Process all videos in the specified folder, extracting every Nth frame.
    Args:
        video_folder (str): Path to the folder containing videos.
        output_dir (str): Directory to save output results.
        frame_interval (int): Extract and process every Nth frame.
    """
    print("Setting up model...")
    model, transform, device = setup_model()
    os.makedirs(output_dir, exist_ok=True)

    # Supported video extensions
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
                # Convert frame to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Save temp image to pass to process_single_image
                temp_img_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
                pil_image.save(temp_img_path)
                # Process the frame
                success = process_single_image(temp_img_path, model, transform, device, output_dir)
                print(f"  Frame {frame_count}: Processed | Success: {success}")
                saved_frame_count += 1
                # Optionally, remove temp image after processing
                os.remove(temp_img_path)
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
    # process_videos("path/to/your/video_folder", output_dir="your_video_output_dir", frame_interval=100)

    # --- EDIT BELOW AS NEEDED ---
    # Uncomment and set your folder paths

    #Example: process images
    process_images("gaze-dataset", output_dir="heatmaps")

    # Example: process videos
    # process_videos("video-dataset", output_dir="video_heatmaps", frame_interval=100)

    # For demonstration, you can uncomment one of the following lines:
    # process_images("gaze-dataset")
    # process_videos("video-dataset")

    print("Edit the main() function to call process_images() or process_videos() with your folder paths.")

if __name__ == "__main__":
    main()