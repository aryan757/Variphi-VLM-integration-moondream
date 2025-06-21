# VLM Integration - Computer Vision Activity Detection System

## 📋 Overview

This directory contains a comprehensive **Vision Language Model (VLM) Integration System** that leverages the **Moondream AI model** to detect and analyze various activities in images and videos. The system is designed for security, surveillance, and monitoring applications.

## 🛠️ System Architecture

The system uses the **Moondream2** model from Hugging Face, which is a powerful vision-language model capable of understanding and describing visual content. It supports both local inference and API-based processing.

### Model Configurations Available:
- **Local Hugging Face Model**: `vikhyatk/moondream2` (full model)
- **Quantized Model**: `moondream/moondream-2b-2025-04-14-4bit` (optimized for performance)
- **API Model**: Cloud-based inference via Moondream API

## 📁 File Structure & Descriptions

### 🔍 **Activity Detection Scripts**

#### 1. `behavioural_analytics.py` - Aggressive Behavior Detection
- **Purpose**: Detects aggressive behavior, violence, and physical fights
- **Detection Capabilities**:
  - Physical assault and attacks between people
  - Violence and aggressive confrontations
  - Fighting and physical altercations
- **Output**: Creates detection results with bounding boxes and point annotations
- **Output Directories**: 
  - `behavioural_analytics_point_output/` - Point-based detections (blue circles)
  - `behavioural_analytics_detect_output/` - Bounding box detections (red rectangles)

#### 2. `phone_usage.py` - Mobile Device Usage Detection
- **Purpose**: Identifies people using phones or mobile devices
- **Detection Capabilities**:
  - Person holding or using mobile phones
  - Device interaction and usage patterns
  - Location-based identification of users
- **Output Directories**:
  - `phone_usage_point_output/` - Point annotations
  - `phone_usage_detect_output/` - Bounding box detections

#### 3. `fire_smoke.py` - Fire and Smoke Detection
- **Purpose**: Detects fire and smoke in images/videos
- **Detection Capabilities**:
  - Fire detection in various environments
  - Smoke identification and analysis
  - Emergency situation monitoring
- **Output Directories**:
  - `fire_smoke_point_output/` - Point-based detections
  - `fire_smoke_detect_output/` - Bounding box detections

#### 4. `theft_or_shoplifting.py` - Theft Detection System
- **Purpose**: Identifies theft and shoplifting activities
- **Detection Capabilities**:
  - Secretive item taking behaviors
  - Shoplifting in retail environments
  - Stealing from individuals or stores
  - Item hiding and concealment detection
- **Output Directories**:
  - `theft_or_shoplifting_point_output/` - Point annotations
  - `theft_or_shoplifting_detect_output/` - Bounding box detections

#### 5. `property_tampering.py` - Vandalism Detection
- **Purpose**: Detects property tampering and vandalism
- **Detection Capabilities**:
  - Property damage and vandalism
  - Unauthorized tampering with objects
  - Destructive behavior identification
- **Output Directories**:
  - `property_tampering_point_output/` - Point annotations
  - `property_tampering_detect_output/` - Bounding box detections

#### 6. `waste_spillage.py` - Environmental Monitoring
- **Purpose**: Detects waste and spillage incidents
- **Detection Capabilities**:
  - Waste accumulation detection
  - Liquid spillage identification
  - Environmental cleanliness monitoring
- **Output Directories**:
  - `spillage_point_output/` - Point-based detections
  - `spillage_detect_output/` - Bounding box detections

#### 7. `work_at_height_edge_dangerous.py` - Safety Monitoring
- **Purpose**: Monitors dangerous work conditions
- **Detection Capabilities**:
  - Work at height safety violations
  - Edge and dangerous area monitoring
  - Safety protocol compliance checking

---

### 📦 **Data Files**

#### ZIP Archives (Dataset Collections):
- **`fire.zip`** - Fire detection training/testing dataset
- **`construction.zip`** - Construction site monitoring data
- **`Data.zip`** - General purpose dataset collection
- **`CCTV_real_data.zip`** - Real-world CCTV footage for testing

---

### ⚙️ **Configuration Files**

#### `requirements.txt` - Python Dependencies
Contains all required Python packages:
```
fastapi          # Web framework for API development
uvicorn[standard] # ASGI server
pillow           # Image processing library
python-multipart # File upload handling
moondream        # VLM model package
requests         # HTTP client library
python-dotenv    # Environment variable management
opencv-python    # Computer vision library
numpy            # Numerical computing
transformers     # Hugging Face transformers
torch            # PyTorch deep learning framework
einops           # Tensor operations
pyvips-binary    # Image processing (binary)
pyvips           # Image processing
accelerate       # Model acceleration
```

---

## 🚀 **Getting Started**

### 1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Model Configuration**
Edit any detection script to choose your preferred model configuration:

```python
# Option 1: Local Hugging Face Model (Recommended)
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True, 
    device_map={"": "mps"},  # Use "cuda" for NVIDIA GPUs, "cpu" for CPU
)

# Option 2: Quantized Model (Faster inference)
model = AutoModelForCausalLM.from_pretrained(
    "moondream/moondream-2b-2025-04-14-4bit",
    trust_remote_code=True,
    device_map={"": "mps"}
)

# Option 3: API Model (Requires API key)
model = md.vl(api_key=MOONDREAM_API_KEY)
```

### 3. **Basic Usage**

#### **Process Images:**
```python
# Example: Detect theft in images
python theft_or_shoplifting.py

# Or modify the main() function to process your specific folder:
process_images("path/to/your/images", output_dir="results")
```

#### **Process Videos:**
```python
# Process videos with frame extraction
process_videos("path/to/videos", output_dir="video_results", frame_interval=100)
```

---

## 🎯 **Core Functionality**

### **MoondreamService Class**
Each detection script contains a `MoondreamService` class with three main methods:

#### 1. **`get_present_activities(image)`**
- Analyzes the image using natural language prompts
- Returns description of detected activities or `None` if nothing found
- Uses specific prompts tailored for each activity type

#### 2. **`point_activities(image, description)`**
- Creates point-based annotations (blue circles) on detected activities
- Saves results with `detected_[random].jpg` filename
- Returns the filename of the annotated image

#### 3. **`detect_activities(image, description)`**
- Creates bounding box annotations (red rectangles) around detected objects
- Provides more precise localization than point detection
- Saves results in respective output directories

---

## 📊 **Output Formats**

### **Detection Results:**
- **Point Detection**: Blue circular markers indicating activity locations
- **Bounding Box Detection**: Red rectangular boxes around detected objects/people
- **Console Output**: Detailed logging of detection results and processing status

### **File Naming Convention:**
- Format: `detected_[random_number].jpg`
- Example: `detected_573.jpg`

---

## 🔧 **Customization**

### **Modify Detection Parameters:**
1. **Frame Interval**: Change `frame_interval` parameter for video processing frequency
2. **Detection Prompts**: Edit the prompt strings in `get_present_activities()` methods
3. **Output Directories**: Modify folder names in each script
4. **Visual Annotations**: Adjust circle radius, box thickness, and colors

### **Adding New Activity Types:**
1. Copy an existing detection script
2. Modify the detection prompt in `get_present_activities()`
3. Update output directory names
4. Customize the main() function for your specific use case

---

## 🖥️ **Hardware Requirements**

### **Minimum Requirements:**
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space for models
- **Processor**: Multi-core CPU (GPU recommended)

### **Recommended Setup:**
- **GPU**: NVIDIA GPU with CUDA support or Apple Silicon with MPS
- **RAM**: 16GB or higher
- **Python**: 3.8 or higher

---

## 🔒 **Security & Privacy**

- **Local Processing**: All inference can be done locally without internet connection
- **Data Privacy**: Images and videos are processed locally by default
- **API Option**: Cloud processing available but requires API keys
- **Output Security**: Results saved locally in designated output directories

---

## 🐛 **Troubleshooting**

### **Common Issues:**

1. **CUDA/MPS Device Errors:**
   - Change `device_map={"": "cpu"}` if GPU is not available

2. **Memory Issues:**
   - Use the quantized model for lower memory usage
   - Reduce batch size or frame processing frequency

3. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

4. **Model Download Issues:**
   - Check internet connection for initial model download
   - Verify Hugging Face model names are correct

---

## 📈 **Performance Tips**

1. **Use GPU acceleration** when available (CUDA/MPS)
2. **Compile models** for repeated inference: `model.model.compile()`
3. **Batch processing** for multiple images
4. **Adjust frame intervals** for video processing based on requirements
5. **Use quantized models** for faster inference with slightly reduced accuracy

---

## 📝 **License & Attribution**

This system uses the Moondream2 model from Hugging Face. Please refer to the original model's license and terms of use for commercial applications.

---

## 🤝 **Support**

For technical support or questions:
1. Check the troubleshooting section above
2. Review the Moondream documentation
3. Verify all dependencies are correctly installed
4. Ensure proper model configuration for your hardware

---

**Happy Detecting! 🎉** 