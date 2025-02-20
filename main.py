from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

# Load model and processor (loaded once at startup)
processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")
model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")

# Define label columns (class names)
label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

# Improved Home Page Endpoint with detailed model information and related resources
@app.get("/", response_class=HTMLResponse)
async def home():
    content = """
    <html>
      <head>
        <title>Chest X-ray Image Classifier API</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; color: #333; line-height: 1.6; }
          header { text-align: center; padding: 20px; }
          section { background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
          code { background: #eee; padding: 2px 4px; border-radius: 4px; }
          pre { background: #eee; padding: 10px; border-radius: 4px; }
          a { color: #0077cc; text-decoration: none; }
          a:hover { text-decoration: underline; }
        </style>
      </head>
      <body>
        <header>
          <h1>Chest X-ray Image Classifier</h1>
          <p>A fine-tuned Vision Transformer (ViT) model for classifying chest radiographs.</p>
        </header>
        
        <section>
          <h2>Model Overview</h2>
          <p>
            This API is powered by a fine-tuned <strong>Vision Transformer</strong> model based on 
            <code>google/vit-base-patch16-224-in21k</code>. It has been adapted for medical image classification using the CheXpert dataset.
            The model distinguishes among five classes: <em>Cardiomegaly, Edema, Consolidation, Pneumonia,</em> and <em>No Finding</em>. 
            The model weights can be downloaded from <a href="https://huggingface.co/codewithdark/vit-chest-xray" target="_blank"> HuggingFace</a>.
          </p>
          <p>
            <strong>Performance:</strong><br>
            Final Validation Accuracy: <span style="color: green;">98.46%</span><br>
            Final Training Loss: 0.1069<br>
            Final Validation Loss: 0.0980
          </p>
        </section>
        
        <section>
          <h2>Dataset & Training Details</h2>
          <p>
            The model is fine-tuned on the <a href="https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2" target="_blank">CheXpert dataset</a>, a large-scale collection of chest X-ray images with multi-label annotations.
            Training was performed using the AdamW optimizer with a learning rate of 3e-5, a batch size of 32, and mixed precision (via <code>torch.amp</code>) over 10 epochs.
          </p>
        </section>
        
        <section>
          <h2>Usage Instructions</h2>
          <p>
            To classify a chest X-ray image, send a POST request to the <code>/predict</code> endpoint with an image file.
          </p>
          <p>
            Example using <code>curl</code>:
          </p>
          <pre>
curl -X POST "http://localhost:8000/predict" -F "image_file=@your_image.jpg"
          </pre>
        </section>
      </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict")
async def predict(image_file: UploadFile = File(...)):
    # Read image file into memory
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Ensure the image is in RGB mode (most image classification models require RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        print("Image converted to RGB.")

    # Preprocess the image using the processor
    inputs = processor(images=image, return_tensors="pt")

    # Make a prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and compute probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).tolist()[0]
    probabilities = {label_columns[i]: probabilities[i] for i in range(len(label_columns))}

    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    predicted_class_label = label_columns[predicted_class_idx]

    # Return the prediction results in JSON format
    return JSONResponse(content={
        "predicted_class_idx": predicted_class_idx,
        "predicted_class_label": predicted_class_label,
        "probabilities": probabilities
    })

# To run the app:
# uvicorn your_filename:app --reload
