# Complete Testing Guide for Health Universe FastAPI Application

## 📋 Overview

This comprehensive test suite validates all Health Universe compliance features and ensures your chest X-ray classifier is ready for production deployment. The tests cover functionality, security, healthcare compliance, performance, and integration scenarios.

## 🚀 Quick Start

### 1. Automated Setup (Recommended)
```bash
# Make setup script executable
chmod +x setup_tests.sh

# Run automated setup and tests
./setup_tests.sh
```

### 2. Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r test-requirements.txt

# Generate test data
python test_data_generator.py

# Run tests
python run_tests.py
```

## 🧪 Test Categories

### 1. **Basic Functionality Tests** (`TestHealthCheck`, `TestHomeEndpoint`)
- ✅ Health check endpoint accessibility
- ✅ Response structure validation
- ✅ System metrics reporting
- ✅ Home page HTML content

**Run Command:**
```bash
python run_tests.py --category basic
```

### 2. **Security & Validation Tests** (`TestInputValidation`, `TestSecurityMiddleware`)
- 🔒 File upload validation (size, type, format)
- 🔒 CORS middleware configuration
- 🔒 Security headers verification
- 🔒 Invalid input rejection

**Run Command:**
```bash
python run_tests.py --category security
```

### 3. **Healthcare Compliance Tests** (`TestLogging`, `TestComplianceFeatures`)
- 🏥 HIPAA-compliant audit logging
- 🏥 Structured logging format
- 🏥 Request ID tracking
- 🏥 Model version tracking
- 🏥 Timestamp compliance

**Run Command:**
```bash
python run_tests.py --category compliance
```

### 4. **Prediction Endpoint Tests** (`TestPredictionEndpoint`)
- 🧠 Model loading verification
- 🧠 Prediction accuracy testing
- 🧠 Response format validation
- 🧠 Error handling scenarios

**Run Command:**
```bash
python run_tests.py --category prediction
```

### 5. **Performance Tests** (`TestPerformance`, `TestLoadAndStress`)
- ⚡ Response time requirements
- ⚡ Memory usage monitoring
- ⚡ Concurrent request handling
- ⚡ Load testing scenarios

**Run Command:**
```bash
python run_tests.py --category performance
```

### 6. **Integration Tests** (`TestIntegrationScenarios`)
- 🔗 End-to-end workflow testing
- 🔗 Component integration verification
- 🔗 Configuration validation

**Run Command:**
```bash
python run_tests.py --category integration
```

## 📊 Test Reports

### Coverage Report
```bash
# Generate HTML coverage report
python -m pytest test_main.py --cov=main --cov-report=html

# View report
open htmlcov/index.html  # On macOS
# OR navigate to htmlcov/index.html in your browser
```

### Detailed JSON Report
After running tests, check `test_results.json` for detailed results:
```json
{
  "timestamp": "2025-07-03 12:00:00",
  "duration_seconds": 45.67,
  "results": {
    "health_check": {"success": true, "output": "..."},
    "security": {"success": true, "output": "..."}
  }
}
```

## 🔍 Test Data Generation

### Synthetic Test Images
The test suite includes a sophisticated test data generator that creates realistic chest X-ray simulations:

```bash
# Generate test dataset
python test_data_generator.py
```

**Generated Test Cases:**
- **Normal chest X-ray** - Baseline case
- **Cardiomegaly simulation** - Enlarged heart
- **Edema pattern** - Pulmonary edema
- **Consolidation area** - Lung consolidation
- **Pneumonia infiltrate** - Infection pattern

### Edge Cases
- Very small/large images
- Different color modes (RGB, Grayscale, etc.)
- Corrupted files
- Invalid file types
- Large file sizes

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```bash
# Check internet connectivity
ping huggingface.co

# Test model download manually
python -c "from transformers import AutoImageProcessor; AutoImageProcessor.from_pretrained('codewithdark/vit-chest-xray')"
```

#### 2. Memory Issues
```bash
# Monitor memory usage during tests
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Run tests with memory profiling
python -m memory_profiler run_tests.py
```

#### 3. Permission Errors
```bash
# Fix script permissions
chmod +x setup_tests.sh

# Check file permissions
ls -la *.py
```

#### 4. Dependency Conflicts
```bash
# Clean environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Test Debugging

#### Enable Verbose Logging
```bash
# Run with detailed output
python run_tests.py --verbose

# Run specific test with debugging
python -m pytest test_main.py::TestHealthCheck::test_health_check_endpoint_exists -v -s
```

#### Mock Model for Testing
```python
# In test files, mock the model to avoid loading issues
@patch('main.model_loaded', True)
@patch('main.model')
@patch('main.processor')
def test_with_mocked_model(mock_processor, mock_model):
    # Your test code here
    pass
```

## 📈 Performance Benchmarking

### Response Time Requirements
- **Health check**: < 1 second
- **Prediction**: < 5 seconds (with model loaded)
- **Home page**: < 500ms

### Memory Requirements
- **Base application**: < 500MB
- **With model loaded**: < 2GB
- **Peak usage**: < 3GB

### Load Testing
```bash
# Install locust for load testing
pip install locust

# Create load test script (example)
# Then run: locust -f load_test.py --host http://localhost:8000
```

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
# .github/workflows/test.yml
name: Health Universe Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Run tests
      run: |
        pip install -r requirements.txt
        pip install -r test-requirements.txt
        python run_tests.py --skip-slow
```

### Local Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
# Run tests before each commit
pre-commit install
```

## ✅ Health Universe Deployment Checklist

Before deploying to Health Universe, ensure all tests pass:

### Required Tests Passing:
- [ ] Health check endpoint responds correctly
- [ ] Input validation works properly
- [ ] HIPAA-compliant logging implemented
- [ ] Error handling doesn't expose sensitive data
- [ ] Model loading and prediction work
- [ ] Performance requirements met
- [ ] Security middleware configured

### Verification Commands:
```bash
# Run complete test suite
python run_tests.py

# Check specific compliance features
python -m pytest test_main.py::TestComplianceFeatures -v

# Verify health endpoint
curl http://localhost:8000/health
```

## 📚 Additional Resources

### Test Documentation
- **pytest**: https://docs.pytest.org/
- **FastAPI Testing**: https://fastapi.tiangolo.com/tutorial/testing/
- **Health Universe Docs**: https://docs.healthuniverse.com/

### Healthcare Compliance
- **HIPAA Security**: Understanding audit logging requirements
- **Medical Device Testing**: FDA guidelines for AI medical devices
- **Clinical Validation**: Best practices for medical AI testing

### Performance Optimization
- **FastAPI Performance**: Async best practices
- **Model Optimization**: Techniques for faster inference
- **Memory Management**: Preventing memory leaks in long-running services

## 🎯 Success Criteria

Your application is ready for Health Universe deployment when:

1. **All tests pass** (100% success rate)
2. **Coverage > 80%** (preferably > 90%)
3. **Response times meet requirements**
4. **No security vulnerabilities detected**
5. **HIPAA compliance features verified**
6. **Model predictions work correctly**

## 🚀 Final Steps

Once all tests pass:

1. **Push to GitHub** with updated code
2. **Deploy via Health Universe interface**
3. **Test on Health Universe platform**
4. **Configure production settings**
5. **Monitor application metrics**

Your chest X-ray classifier is now thoroughly tested and ready for secure, compliant deployment on Health Universe! 🎉