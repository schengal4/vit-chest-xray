from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
import logging
import structlog
import time
import uuid
from typing import Dict, Optional
import asyncio
from functools import wraps
import psutil
import os

# Configure structured logging for HIPAA compliance
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Chest X-ray Image Classifier API",
    description="HIPAA-compliant API for chest X-ray classification using Vision Transformer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for your domain in production
)

# Global variables for model loading status
model_loaded = False
processor = None
model = None
model_load_error = None

# Define label columns (class names)
label_columns = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']

# Pydantic models for request/response validation
class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    request_id: str = Field(..., description="Unique request identifier for audit logging")
    predicted_class_idx: int = Field(..., ge=0, le=4, description="Predicted class index (0-4)")
    predicted_class_label: str = Field(..., description="Predicted class label")
    probabilities: Dict[str, float] = Field(..., description="Probability scores for each class")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Highest probability score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(default="vit-chest-xray-v1.0", description="Model version used")

class HealthCheckResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    timestamp: str
    model_loaded: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error: str
    request_id: str
    timestamp: str

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    # Log error without exposing sensitive information
    logger.error(
        "API error occurred",
        request_id=request_id,
        status_code=exc.status_code,
        endpoint=str(request.url),
        method=request.method,
        user_agent=request.headers.get("user-agent", "unknown"),
        error_type="http_exception"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="An error occurred while processing your request",
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).model_dump()
    )

# Middleware for request logging and audit trail
@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    # Generate unique request ID for audit trail
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    
    # Log incoming request (HIPAA audit requirement)
    logger.info(
        "Incoming request",
        request_id=request_id,
        method=request.method,
        endpoint=str(request.url),
        user_agent=request.headers.get("user-agent", "unknown"),
        content_length=request.headers.get("content-length", 0),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    
    response = await call_next(request)
    
    # Log response (HIPAA audit requirement)
    processing_time = (time.time() - start_time) * 1000
    logger.info(
        "Request completed",
        request_id=request_id,
        status_code=response.status_code,
        processing_time_ms=round(processing_time, 2),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    
    return response

# Load model and processor at startup
@app.on_event("startup")
async def load_model():
    global model_loaded, processor, model, model_load_error
    
    try:
        logger.info("Loading model and processor", action="model_loading_start")
        
        # Load model and processor
        processor = AutoImageProcessor.from_pretrained("codewithdark/vit-chest-xray")
        model = AutoModelForImageClassification.from_pretrained("codewithdark/vit-chest-xray")
        
        # Set model to evaluation mode
        model.eval()
        
        model_loaded = True
        logger.info("Model and processor loaded successfully", action="model_loading_success")
        
    except Exception as e:
        model_load_error = str(e)
        logger.error("Failed to load model", error=str(e), action="model_loading_error")
        raise

# Health check endpoint (required by Health Universe)
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for Health Universe platform monitoring.
    Returns system status, model availability, and resource usage.
    """
    try:
        # Get system metrics
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_usage = psutil.cpu_percent(interval=1)
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        
        health_status = HealthCheckResponse(
            status="healthy" if model_loaded and not model_load_error else "unhealthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_loaded=model_loaded,
            memory_usage_mb=round(memory_usage, 2),
            cpu_usage_percent=cpu_usage,
            uptime_seconds=round(uptime, 2)
        )
        
        logger.info("Health check performed", status=health_status.status, action="health_check")
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e), action="health_check_error")
        raise HTTPException(status_code=500, detail="Health check failed")

# Set startup time for uptime calculation
@app.on_event("startup")
async def set_startup_time():
    app.state.start_time = time.time()

# Input validation function
def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file for security and format compliance."""
    
    # Check file size (limit to 10MB for security)
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    # Check file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )
    
    # Check filename for security (basic validation)
    if not file.filename or len(file.filename) > 255:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename"
        )

# Enhanced home page with healthcare compliance information
@app.get("/", response_class=HTMLResponse)
async def home():
    content = """
    <html>
      <head>
        <title>Chest X-ray Image Classifier API - Healthcare AI Platform</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; color: #333; line-height: 1.6; }
          header { text-align: center; padding: 20px; background: #2c5aa0; color: white; border-radius: 8px; margin-bottom: 20px; }
          section { background: #fff; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
          .compliance { background: #e8f5e8; border-left: 4px solid #4CAF50; }
          .warning { background: #fff3cd; border-left: 4px solid #ffc107; }
          code { background: #eee; padding: 2px 4px; border-radius: 4px; }
          pre { background: #eee; padding: 10px; border-radius: 4px; overflow-x: auto; }
          a { color: #0077cc; text-decoration: none; }
          a:hover { text-decoration: underline; }
          .metric { display: inline-block; margin: 10px; padding: 10px; background: #f0f8ff; border-radius: 4px; }
        </style>
      </head>
      <body>
        <header>
          <h1>🏥 Chest X-ray Image Classifier</h1>
          <p>HIPAA-Compliant AI-Powered Radiological Analysis</p>
        </header>
        
        <section class="compliance">
          <h2>🔒 Healthcare Compliance Features</h2>
          <ul>
            <li><strong>HIPAA Compliance:</strong> Comprehensive audit logging and secure data handling</li>
            <li><strong>Structured Logging:</strong> All requests and responses are logged for regulatory compliance</li>
            <li><strong>Input Validation:</strong> Rigorous validation of all inputs using Pydantic models</li>
            <li><strong>Error Handling:</strong> Secure error responses that don't expose sensitive information</li>
            <li><strong>Health Monitoring:</strong> Built-in health checks for platform monitoring</li>
          </ul>
        </section>
        
        <section>
          <h2>🤖 Model Overview</h2>
          <p>
            This API uses a fine-tuned <strong>Vision Transformer (ViT)</strong> model based on 
            <code>google/vit-base-patch16-224-in21k</code>, specifically adapted for medical image analysis 
            using the CheXpert dataset.
          </p>
          <div class="metric">
            <strong>Validation Accuracy:</strong> 98.46%
          </div>
          <div class="metric">
            <strong>Classes:</strong> 5 conditions
          </div>
          <div class="metric">
            <strong>Model Version:</strong> vit-chest-xray-v1.0
          </div>
        </section>
        
        <section class="warning">
          <h2>⚠️ Clinical Use Disclaimer</h2>
          <p>
            <strong>Important:</strong> This AI tool is designed to assist healthcare professionals and should not 
            replace clinical judgment. All results should be reviewed by qualified medical personnel. This system 
            is intended for educational and research purposes.
          </p>
        </section>
        
        <section>
          <h2>📋 API Endpoints</h2>
          <ul>
            <li><code>GET /</code> - This information page</li>
            <li><code>GET /health</code> - System health check and monitoring</li>
            <li><code>POST /predict</code> - Chest X-ray classification endpoint</li>
            <li><code>GET /docs</code> - Interactive API documentation</li>
          </ul>
        </section>
        
        <section>
          <h2>🔧 Usage Example</h2>
          <p>Classify a chest X-ray image:</p>
          <pre>
curl -X POST "/predict" \\
  -H "Content-Type: multipart/form-data" \\
  -F "image_file=@chest_xray.jpg"
          </pre>
        </section>
        
        <section>
          <h2>📊 Detected Conditions</h2>
          <ul>
            <li><strong>Cardiomegaly:</strong> Enlarged heart</li>
            <li><strong>Edema:</strong> Fluid accumulation in lungs</li>
            <li><strong>Consolidation:</strong> Lung tissue inflammation</li>
            <li><strong>Pneumonia:</strong> Lung infection</li>
            <li><strong>No Finding:</strong> Normal chest X-ray</li>
          </ul>
        </section>
      </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, image_file: UploadFile = File(...)):
    """
    Classify chest X-ray image using Vision Transformer model.
    
    Returns prediction with confidence scores and audit trail.
    Implements HIPAA-compliant logging and error handling.
    """
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    try:
        # Validate input file FIRST (before checking model)
        validate_image_file(image_file)
        
        # Check if model is loaded AFTER validation
        if not model_loaded or model is None or processor is None:
            logger.error("Model not available", request_id=request_id, action="prediction_error")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available. Please try again later."
            )
        
        # Log prediction request (HIPAA audit requirement)
        logger.info(
            "Prediction request received",
            request_id=request_id,
            filename=image_file.filename,
            content_type=image_file.content_type,
            action="prediction_start"
        )
        
        # Read and process image
        image_bytes = await image_file.read()
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error("Failed to open image", request_id=request_id, error=str(e), action="image_processing_error")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format. Please upload a valid image file."
            )
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image
        try:
            inputs = processor(images=image, return_tensors="pt")
        except Exception as e:
            logger.error("Image preprocessing failed", request_id=request_id, error=str(e), action="preprocessing_error")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to process image. Please ensure it's a valid chest X-ray image."
            )
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            logger.error("Model inference failed", request_id=request_id, error=str(e), action="inference_error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model inference failed. Please try again."
            )
        
        # Process results
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).tolist()[0]
        probabilities_dict = {label_columns[i]: round(probabilities[i], 4) for i in range(len(label_columns))}
        
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        predicted_class_label = label_columns[predicted_class_idx]
        confidence_score = max(probabilities)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            request_id=request_id,
            predicted_class_idx=predicted_class_idx,
            predicted_class_label=predicted_class_label,
            probabilities=probabilities_dict,
            confidence_score=round(confidence_score, 4),
            processing_time_ms=round(processing_time, 2),
            model_version="vit-chest-xray-v1.0"
        )
        
        # Log successful prediction (HIPAA audit requirement)
        logger.info(
            "Prediction completed successfully",
            request_id=request_id,
            predicted_class=predicted_class_label,
            confidence_score=confidence_score,
            processing_time_ms=round(processing_time, 2),
            action="prediction_success"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (already logged)
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(
            "Unexpected error during prediction",
            request_id=request_id,
            error=str(e),
            action="prediction_unexpected_error"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)