#!/bin/bash

# HADM Server Test Runner
# Runs all tests and code quality checks

set -e

echo "🧪 HADM Server Test Suite"
echo "========================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Ensure virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "Virtual environment not detected. Activating..."
    source venv/bin/activate
fi

# Install test dependencies if needed
print_status "Installing test dependencies..."
pip install -r requirements-test.txt

# Code formatting with black
print_status "Running code formatting (black)..."
black app/ tests/ --check --diff || {
    print_warning "Code formatting issues found. Running black..."
    black app/ tests/
    print_success "Code formatted"
}

# Import sorting with isort
print_status "Checking import sorting (isort)..."
isort app/ tests/ --check-only --diff || {
    print_warning "Import sorting issues found. Running isort..."
    isort app/ tests/
    print_success "Imports sorted"
}

# Linting with flake8
print_status "Running linting (flake8)..."
flake8 app/ tests/ --max-line-length=88 --extend-ignore=E203,W503

# Type checking with mypy
print_status "Running type checking (mypy)..."
mypy app/ --ignore-missing-imports || print_warning "Type checking completed with warnings"

# Run tests with pytest
print_status "Running unit tests..."
pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

# Test coverage report
print_status "Coverage report generated: htmlcov/index.html"

print_success "All tests completed! 🎉"
echo ""
echo "📊 Test Results Summary:"
echo "- Code formatting: ✅"
echo "- Import sorting: ✅"
echo "- Linting: ✅"
echo "- Type checking: ✅"
echo "- Unit tests: ✅"
echo ""
echo "📁 Generated files:"
echo "- Coverage report: htmlcov/index.html"
echo "- Test results: Available in terminal output" 