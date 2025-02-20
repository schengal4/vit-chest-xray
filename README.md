# Chest X-ray Image Classifier API

This repository provides a REST API built with FastAPI that leverages a fine-tuned Vision Transformer (ViT) model for classifying chest X-ray images. The model is adapted from the [codewithdark/vit-chest-xray](https://huggingface.co/codewithdark/vit-chest-xray) repository and has been refined on the CheXpert dataset to identify multiple lung conditions.

## Model Overview

The model is based on the [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) architecture and is fine-tuned to distinguish among five classes:
- **Cardiomegaly**
- **Edema**
- **Consolidation**
- **Pneumonia**
- **No Finding**

Key performance highlights include:
- **Final Validation Accuracy:** 98.46%
- **Final Training Loss:** 0.1069
- **Final Validation Loss:** 0.0980

By utilizing the attention mechanism inherent to Vision Transformers, the model effectively extracts and analyzes features from chest radiographs.

## Dataset

The model is fine-tuned on the [CheXpert dataset](https://stanfordmlgroup.github.io/chexpert/), a large-scale collection of chest X-ray images with multi-label annotations. This dataset includes both frontal and lateral views, providing comprehensive data for detecting various lung abnormalities.

## Training Details

The fine-tuning process involved:
- **Optimizer:** AdamW
- **Learning Rate:** 3e-5
- **Batch Size:** 32
- **Epochs:** 10
- **Loss Function:** Binary Cross-Entropy with Logits
- **Precision:** Mixed precision training using `torch.amp`

These parameters were optimized to ensure robust performance and generalization to unseen data.

## API Usage

The API is built with FastAPI and exposes the following endpoints:

### Endpoints

- **GET /**: Returns an HTML homepage with model details and usage instructions.
- **POST /predict**: Accepts an image file and returns the predicted class index, label, and corresponding probabilities.

### Example with cURL

```bash
curl -X POST "http://localhost:8000/predict" -F "image_file=@your_image.jpg"
```

This command sends an image to the `/predict` endpoint and returns a JSON response similar to:

```json
{
  "predicted_class_idx": 4,
  "predicted_class_label": "No Finding",
  "probabilities": {
    "Cardiomegaly": 0.01,
    "Edema": 0.02,
    "Consolidation": 0.03,
    "Pneumonia": 0.05,
    "No Finding": 0.89
  }
}
```

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/chest-xray-classifier.git
   cd chest-xray-classifier
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI server:**

   ```bash
   uvicorn main:app --reload
   ```

4. **Access the homepage:**  
   Open your browser and navigate to `http://localhost:8000/` to view API details and usage instructions.

## Fine-Tuning the Model

To fine-tune the model on your own dataset:
1. Adapt the image preprocessing steps (ensuring images are in RGB format).
2. Use the `AutoImageProcessor` and `AutoModelForImageClassification` from the `transformers` library to load the model.
3. Modify hyperparameters (learning rate, batch size, epochs, etc.) as needed.

Refer to the source code in this repository for detailed instructions on adjusting the training pipeline.

## Contributing

Contributions are welcome! If you have suggestions, improvements, or bug fixes, please fork the repository and submit a pull request. Follow standard coding and testing practices.

## Credit & Acknowledgements

- **Initial Model Source:** This project builds upon the work available in [codewithdark/vit-chest-xray](https://huggingface.co/codewithdark/vit-chest-xray). All credit for the original model and its fine-tuning methodology goes to codewithdark.
- **CheXpert Dataset:** Thanks to the Stanford ML Group for providing the CheXpert dataset.
- **Hugging Face:** For the `transformers` library and the model hub which greatly simplify model sharing and fine-tuning.
- **Community Contributions:** Special thanks to everyone who has contributed ideas and improvements.

## License

This project is available under the [MIT License](LICENSE).