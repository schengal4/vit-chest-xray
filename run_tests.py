#!/usr/bin/env python3
"""
Comprehensive test runner for Health Universe FastAPI application.
This script runs all tests and generates detailed reports for healthcare compliance verification.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class HealthUniverseTestRunner:
    """Test runner for Health Universe compliance verification."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and capture output."""
        print(f"\n🔍 {description}")
        print(f"Running: {' '.join(command)}")
        print("-" * 60)
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success:
                print(f"✅ {description} - PASSED")
            else:
                print(f"❌ {description} - FAILED")
                print(f"Exit code: {result.returncode}")
            
            if output.strip():
                print("Output:")
                print(output)
            
            return success, output
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - TIMEOUT")
            return False, "Test timed out after 5 minutes"
        except Exception as e:
            print(f"💥 {description} - ERROR: {str(e)}")
            return False, str(e)
    
    def run_basic_tests(self) -> bool:
        """Run basic functionality tests."""
        print("\n" + "="*60)
        print("🧪 RUNNING BASIC FUNCTIONALITY TESTS")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestHealthCheck", "-v"],
            "Health Check Endpoint Tests"
        )
        self.test_results['health_check'] = {'success': success, 'output': output}
        
        success2, output2 = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestHomeEndpoint", "-v"],
            "Home Page Endpoint Tests"
        )
        self.test_results['home_endpoint'] = {'success': success2, 'output': output2}
        
        return success and success2
    
    def run_security_tests(self) -> bool:
        """Run security and validation tests."""
        print("\n" + "="*60)
        print("🔒 RUNNING SECURITY & VALIDATION TESTS")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestInputValidation", "-v"],
            "Input Validation Security Tests"
        )
        self.test_results['input_validation'] = {'success': success, 'output': output}
        
        success2, output2 = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestSecurityMiddleware", "-v"],
            "Security Middleware Tests"
        )
        self.test_results['security_middleware'] = {'success': success2, 'output': output2}
        
        success3, output3 = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestErrorHandling", "-v"],
            "Error Handling Security Tests"
        )
        self.test_results['error_handling'] = {'success': success3, 'output': output3}
        
        return success and success2 and success3
    
    def run_compliance_tests(self) -> bool:
        """Run healthcare compliance tests."""
        print("\n" + "="*60)
        print("🏥 RUNNING HEALTHCARE COMPLIANCE TESTS")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestLogging", "-v"],
            "Audit Logging Compliance Tests"
        )
        self.test_results['logging'] = {'success': success, 'output': output}
        
        success2, output2 = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestComplianceFeatures", "-v"],
            "Healthcare Compliance Features Tests"
        )
        self.test_results['compliance_features'] = {'success': success2, 'output': output2}
        
        return success and success2
    
    def run_prediction_tests(self) -> bool:
        """Run prediction endpoint tests."""
        print("\n" + "="*60)
        print("🧠 RUNNING PREDICTION ENDPOINT TESTS")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestPredictionEndpoint", "-v"],
            "Prediction Endpoint Functionality Tests"
        )
        self.test_results['prediction_endpoint'] = {'success': success, 'output': output}
        
        return success
    
    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        print("\n" + "="*60)
        print("⚡ RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestPerformance", "-v"],
            "Performance Requirements Tests"
        )
        self.test_results['performance'] = {'success': success, 'output': output}
        
        success2, output2 = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestLoadAndStress", "-v", "-m", "not slow"],
            "Load Testing (Fast Tests Only)"
        )
        self.test_results['load_testing'] = {'success': success2, 'output': output2}
        
        return success and success2
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("\n" + "="*60)
        print("🔗 RUNNING INTEGRATION TESTS")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestIntegrationScenarios", "-v"],
            "End-to-End Integration Tests"
        )
        self.test_results['integration'] = {'success': success, 'output': output}
        
        success2, output2 = self.run_command(
            ["python", "-m", "pytest", "test_main.py::TestConfiguration", "-v"],
            "Configuration and Environment Tests"
        )
        self.test_results['configuration'] = {'success': success2, 'output': output2}
        
        return success and success2
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality and style checks."""
        print("\n" + "="*60)
        print("📝 RUNNING CODE QUALITY CHECKS")
        print("="*60)
        
        # Try to run flake8 if available
        try:
            success, output = self.run_command(
                ["python", "-m", "flake8", "main.py", "--max-line-length=120"],
                "Code Style Check (flake8)"
            )
            self.test_results['code_style'] = {'success': success, 'output': output}
        except:
            print("⚠️  flake8 not available, skipping code style check")
            self.test_results['code_style'] = {'success': True, 'output': 'Skipped - flake8 not available'}
            success = True
        
        # Check for basic Python syntax
        success2, output2 = self.run_command(
            ["python", "-m", "py_compile", "main.py"],
            "Python Syntax Check"
        )
        self.test_results['syntax_check'] = {'success': success2, 'output': output2}
        
        return success and success2
    
    def generate_coverage_report(self) -> bool:
        """Generate test coverage report."""
        print("\n" + "="*60)
        print("📊 GENERATING COVERAGE REPORT")
        print("="*60)
        
        success, output = self.run_command(
            ["python", "-m", "pytest", "test_main.py", "--cov=main", "--cov-report=html", "--cov-report=term"],
            "Test Coverage Analysis"
        )
        self.test_results['coverage'] = {'success': success, 'output': output}
        
        if success:
            print("\n📁 Coverage report generated in 'htmlcov/' directory")
        
        return success
    
    def run_all_tests(self, skip_slow: bool = False) -> None:
        """Run all test suites."""
        print("🚀 Starting Health Universe FastAPI Test Suite")
        print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Test categories in order of importance
        test_categories = [
            ("Basic Functionality", self.run_basic_tests),
            ("Security & Validation", self.run_security_tests),
            ("Healthcare Compliance", self.run_compliance_tests),
            ("Prediction Endpoint", self.run_prediction_tests),
            ("Performance", self.run_performance_tests),
            ("Integration", self.run_integration_tests),
            ("Code Quality", self.run_code_quality_checks),
        ]
        
        results = {}
        all_passed = True
        
        for category_name, test_function in test_categories:
            try:
                passed = test_function()
                results[category_name] = passed
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"💥 Error in {category_name}: {str(e)}")
                results[category_name] = False
                all_passed = False
        
        # Generate coverage report last
        if not skip_slow:
            print("\n" + "="*60)
            print("📊 GENERATING FINAL COVERAGE REPORT")
            print("="*60)
            self.generate_coverage_report()
        
        # Generate final report
        self.generate_final_report(results, all_passed)
    
    def generate_final_report(self, results: Dict[str, bool], all_passed: bool) -> None:
        """Generate final test report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n" + "="*80)
        print("📋 FINAL TEST REPORT")
        print("="*80)
        
        print(f"⏱️  Total execution time: {duration:.2f} seconds")
        print(f"📅 Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📊 Test Category Results:")
        print("-" * 40)
        
        passed_count = 0
        total_count = len(results)
        
        for category, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{category:<25} {status}")
            if passed:
                passed_count += 1
        
        print("-" * 40)
        print(f"Overall Success Rate: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Your application is ready for Health Universe deployment!")
            print("\n✅ Health Universe Compliance Checklist:")
            print("   ✅ Health check endpoint implemented")
            print("   ✅ HIPAA-compliant audit logging")
            print("   ✅ Input validation and security")
            print("   ✅ Error handling (no sensitive data exposure)")
            print("   ✅ Performance requirements met")
            print("   ✅ Structured logging for compliance")
            
            print("\n🚀 Next Steps:")
            print("   1. Push your code to GitHub")
            print("   2. Deploy via Health Universe interface")
            print("   3. Test on Health Universe platform")
            print("   4. Configure production settings")
            
        else:
            print("\n⚠️  SOME TESTS FAILED - Review the failures above before deployment")
            print("\n🔧 Common Issues to Check:")
            print("   - Ensure all dependencies are installed")
            print("   - Check that model files are accessible")
            print("   - Verify network connectivity for model downloads")
            print("   - Review error messages in failed test output")
        
        # Save detailed results to file
        self.save_test_results()
    
    def save_test_results(self) -> None:
        """Save detailed test results to JSON file."""
        try:
            results_file = "test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_seconds': time.time() - self.start_time,
                    'results': self.test_results
                }, f, indent=2)
            
            print(f"\n📄 Detailed test results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Could not save test results: {str(e)}")


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Health Universe FastAPI Test Suite")
    parser.add_argument("--skip-slow", action="store_true", 
                       help="Skip slow tests (load testing, etc.)")
    parser.add_argument("--category", choices=[
        "basic", "security", "compliance", "prediction", 
        "performance", "integration", "quality"
    ], help="Run only specific test category")
    
    args = parser.parse_args()
    
    runner = HealthUniverseTestRunner()
    
    if args.category:
        # Run specific category
        category_map = {
            "basic": runner.run_basic_tests,
            "security": runner.run_security_tests,
            "compliance": runner.run_compliance_tests,
            "prediction": runner.run_prediction_tests,
            "performance": runner.run_performance_tests,
            "integration": runner.run_integration_tests,
            "quality": runner.run_code_quality_checks,
        }
        
        print(f"🎯 Running {args.category} tests only...")
        success = category_map[args.category]()
        
        if success:
            print(f"✅ {args.category.title()} tests PASSED")
            sys.exit(0)
        else:
            print(f"❌ {args.category.title()} tests FAILED")
            sys.exit(1)
    else:
        # Run all tests
        runner.run_all_tests(skip_slow=args.skip_slow)


if __name__ == "__main__":
    main()