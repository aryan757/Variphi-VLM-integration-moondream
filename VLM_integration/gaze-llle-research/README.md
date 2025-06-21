# Gaze-LLE Research Project

A Python-based gaze detection system that uses the Gaze-LLE (Gaze - Learning to Look Everyone) model to analyze human gaze patterns in images and videos. This project can detect faces, predict gaze directions, and visualize gaze heatmaps with in-frame/out-of-frame gaze scoring.

## 🎯 What This Project Does

- **Face Detection**: Automatically detects human faces in images and videos using RetinaFace
- **Gaze Prediction**: Predicts where each person is looking using the Gaze-LLE model
- **Gaze Visualization**: Creates heatmap overlays showing gaze directions
- **In/Out Frame Analysis**: Determines if a person is looking within or outside the frame
- **Batch Processing**: Processes entire folders of images or videos automatically

## 📁 Project Structure

```
gaze-llle-research/
├── gaze_video_processor.py    # Main processing script
├── requirements.txt           # Python dependencies
├── README.md                 # This documentation
├── venv/                     # Virtual environment (optional)
├── gaze-dataset/            # Input images folder
├── heatmaps/                # Output folder for processed images
└── output_images/           # Additional output folder
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or Apple Silicon Mac (MPS support)
- Sufficient storage for model weights (~2GB)

### 1. Clone and Setup Environment
```bash
# Navigate to the project directory
cd gaze-llle-research

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Download
The Gaze-LLE model will be automatically downloaded on first run via PyTorch Hub.

## 🚀 Usage

### Processing Images
```python
from gaze_video_processor import process_images

# Process all images in a folder
process_images("path/to/your/images", output_dir="results")
```

### Processing Videos
```python
from gaze_video_processor import process_videos

# Process videos, extracting every 100th frame
process_videos("path/to/videos", output_dir="video_results", frame_interval=100)
```

### Command Line Usage
Edit the `main()` function in `gaze_video_processor.py` and run:
```bash
python gaze_video_processor.py
```

## 📊 Code Structure Overview

### Core Functions

#### `setup_model()`
- Initializes the Gaze-LLE model
- Detects available hardware (CUDA/MPS/CPU)
- Returns model, transform, and device

#### `process_single_image(image_path, model, transform, device, output_dir)`
- Processes one image for gaze detection
- Detects faces using RetinaFace
- Generates gaze predictions and visualizations
- Saves combined result with bounding boxes and gaze arrows

#### `process_images(image_folder, output_dir)`
- Batch processes all images in a folder
- Supports: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- Creates output directory automatically

#### `process_videos(video_folder, output_dir, frame_interval)`
- Extracts frames from videos at specified intervals
- Processes each frame for gaze detection
- Supports: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

### Visualization Functions

#### `visualize_heatmap(pil_image, heatmap, bbox, inout_score)`
- Creates heatmap overlay for a single person
- Shows gaze probability distribution

#### `visualize_all(pil_image, heatmaps, bboxes, inout_scores)`
- Combined visualization for multiple people
- Shows face bounding boxes with different colors
- Draws gaze direction arrows from face center to gaze target
- Displays in-frame confidence scores

## 🎨 Output Format

For each processed image, the system generates:

### Image Outputs
- `{filename}_combined_result.png`: Final visualization with:
  - Colored bounding boxes around detected faces
  - Gaze direction arrows pointing to predicted gaze targets
  - In-frame confidence scores
  - Gaze target points (circles)

### Visual Elements
- **Bounding Boxes**: Different colors for each person (lime, tomato, cyan, fuchsia, yellow)
- **Gaze Arrows**: Lines from face center to predicted gaze location
- **Gaze Points**: Small circles marking the strongest gaze attention area
- **Confidence Text**: In-frame probability scores

## ⚙️ Configuration Options

### Model Parameters
- **Device Selection**: Automatically chooses CUDA > MPS > CPU
- **In-frame Threshold**: Default 0.5 (adjustable in `visualize_all()`)

### Processing Parameters
- **Frame Interval**: For videos, process every Nth frame (default: 100)
- **Image Extensions**: Configurable in `process_images()`
- **Video Extensions**: Configurable in `process_videos()`

## 🔍 Technical Details

### Dependencies
- **PyTorch**: Deep learning framework
- **Gaze-LLE**: Pre-trained gaze estimation model
- **RetinaFace**: Face detection
- **OpenCV**: Image/video processing
- **PIL/Pillow**: Image manipulation
- **Matplotlib**: Colormap generation
- **NumPy**: Numerical operations

### Model Architecture
- **Gaze-LLE**: Uses DINOv2 ViT-L/14 backbone
- **Input Size**: 448x448 pixels
- **Output**: Gaze heatmaps + in/out frame classification

### Performance Notes
- GPU processing significantly faster than CPU
- Memory usage scales with number of detected faces
- Video processing time depends on frame interval setting

## 🐛 Troubleshooting

### Common Issues

1. **No faces detected**: 
   - Check image quality and lighting
   - Ensure faces are clearly visible
   - Original image saved with `_no_faces.png` suffix

2. **CUDA out of memory**:
   - Reduce batch size or image resolution
   - Use CPU fallback: `device = 'cpu'`

3. **Model download fails**:
   - Check internet connection
   - Verify PyTorch Hub access
   - Clear PyTorch cache: `torch.hub._get_cache_dir()`

4. **Video processing slow**:
   - Increase `frame_interval` parameter
   - Use GPU acceleration
   - Process shorter video segments

## 📝 Example Usage

```python
# Quick start example
if __name__ == "__main__":
    # Process images in current directory
    process_images("gaze-dataset", output_dir="heatmaps")
    
    # Process videos with custom frame interval
    # process_videos("video-dataset", output_dir="video_heatmaps", frame_interval=50)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Test your changes thoroughly
4. Submit a pull request

## 📄 License

This project uses the Gaze-LLE model which has its own licensing terms. Please check the original model's license for commercial use.

## 🙏 Acknowledgments

- [Gaze-LLE Model](https://github.com/fkryan/gazelle) by the original authors
- RetinaFace for robust face detection
- PyTorch team for the deep learning framework 