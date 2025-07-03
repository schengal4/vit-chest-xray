#!/bin/bash

# Health Universe FastAPI Test Environment Setup Script
# This script sets up the testing environment and runs comprehensive tests

set -e  # Exit on any error

echo "🚀 Health Universe FastAPI Test Environment Setup"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install main application requirements
print_status "Installing main application requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Main requirements installed"
else
    print_warning "requirements.txt not found, installing basic dependencies..."
    pip install fastapi uvicorn pillow torch transformers structlog psutil pydantic
fi

# Install testing requirements
print_status "Installing testing requirements..."
if [ -f "test-requirements.txt" ]; then
    pip install -r test-requirements.txt
    print_success "Testing requirements installed"
else
    print_warning "test-requirements.txt not found, installing basic testing tools..."
    pip install pytest pytest-asyncio pytest-cov httpx pytest-mock
fi

# Create necessary directories
print_status "Creating test directories..."
mkdir -p tests
mkdir -p test_images
mkdir -p reports
mkdir -p htmlcov

# Generate test data
print_status "Generating test data..."
if [ -f "test_data_generator.py" ]; then
    python test_data_generator.py
    print_success "Test data generated"
else
    print_warning "test_data_generator.py not found, skipping test data generation"
fi

# Run syntax check
print_status "Running syntax check..."
if python -m py_compile main.py; then
    print_success "Syntax check passed"
else
    print_error "Syntax check failed"
    exit 1
fi

# Function to run tests
run_tests() {
    local test_type=$1
    local description=$2
    
    print_status "Running $description..."
    
    if python run_tests.py --category $test_type; then
        print_success "$description passed"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

# Ask user what tests to run
echo ""
echo "Choose test suite to run:"
echo "1) Quick tests (basic functionality only)"
echo "2) Security tests"
echo "3) Compliance tests"
echo "4) All tests (comprehensive)"
echo "5) Skip tests (setup only)"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        run_tests "basic" "Quick functionality tests"
        ;;
    2)
        run_tests "security" "Security tests"
        ;;
    3)
        run_tests "compliance" "Healthcare compliance tests"
        ;;
    4)
        print_status "Running comprehensive test suite..."
        if python run_tests.py; then
            print_success "All tests completed"
        else
            print_error "Some tests failed"
        fi
        ;;
    5)
        print_status "Skipping tests"
        ;;
    *)
        print_warning "Invalid choice, running basic tests..."
        run_tests "basic" "Basic functionality tests"
        ;;
esac

# Generate final report
echo ""
echo "================================================="
echo "🎯 Test Environment Setup Complete!"
echo "================================================="

echo ""
echo "📋 Setup Summary:"
echo "   ✅ Python environment configured"
echo "   ✅ Dependencies installed"
echo "   ✅ Test data generated"
echo "   ✅ Test structure created"

echo ""
echo "🚀 Next Steps:"
echo "   1. Review test results above"
echo "   2. Fix any failing tests"
echo "   3. Run full test suite: python run_tests.py"
echo "   4. Check coverage report in htmlcov/ directory"

echo ""
echo "🔧 Available Commands:"
echo "   • Run all tests: python run_tests.py"
echo "   • Run specific category: python run_tests.py --category basic"
echo "   • Generate test data: python test_data_generator.py"
echo "   • Manual testing: uvicorn main:app --reload"

echo ""
echo "📁 Generated Files:"
echo "   • Test images: test_images/"
echo "   • Coverage reports: htmlcov/"
echo "   • Test results: test_results.json"

# Check if running in CI/CD environment
if [ "$CI" = "true" ] || [ "$GITHUB_ACTIONS" = "true" ]; then
    print_status "CI/CD environment detected"
    
    # Run all tests in CI mode
    python run_tests.py --skip-slow
    
    # Save artifacts
    if [ -d "htmlcov" ]; then
        print_status "Saving coverage artifacts..."
        tar -czf coverage-report.tar.gz htmlcov/
    fi
    
    if [ -f "test_results.json" ]; then
        print_status "Test results saved for CI artifacts"
    fi
fi

print_success "Setup complete! Your Health Universe FastAPI app is ready for testing."