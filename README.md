# Moondream Variphi API

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your Moondream API key in the `.env` file:
   ```env
   MOONDRM_API_KEY=your_moondream_api_key_here
   ```

3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

- **POST** `/detect-activities/`
  - Upload an image file (form field: `file`).
  - The server will process the image and save the detection result locally.
  - The response will include the path to the saved detection image and detected activities.

### Example (using curl):
```bash
curl -F "file=@/path/to/your/image.jpg" http://localhost:8000/detect-activities/
``` 