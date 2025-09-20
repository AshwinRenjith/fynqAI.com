#!/usr/bin/env python3
"""
Comprehensive test runner for fynqAI backend testing infrastructure.
Provides multiple test execution modes and reporting capabilities.
"""

import argparse
import sys
import subprocess
import time
from typing import List, Dict, Any

# Test configuration
TEST_CONFIGS = {
    "unit": {
        "path": "app/tests/unit/",
        "markers": "",
        "description": "Run unit tests for individual components",
        "coverage_min": 85
    },
    "integration": {
        "path": "app/tests/integration/",
        "markers": "",
        "description": "Run integration tests for workflow validation",
        "coverage_min": 75
    },
    "performance": {
        "path": "app/tests/performance/",
        "markers": "performance",
        "description": "Run performance and load tests",
        "coverage_min": 60
    },
    "api": {
        "path": "app/tests/unit/test_api_v1.py",
        "markers": "",
        "description": "Run API endpoint tests only",
        "coverage_min": 90
    },
    "ai": {
        "path": "app/tests/unit/test_ai_core.py",
        "markers": "",
        "description": "Run AI core module tests only",
        "coverage_min": 80
    },
    "services": {
        "path": "app/tests/unit/test_services.py",
        "markers": "",
        "description": "Run business service tests only",
        "coverage_min": 85
    },
    "workers": {
        "path": "app/tests/unit/test_workers.py",
        "markers": "",
        "description": "Run background worker tests only",
        "coverage_min": 80
    },
    "all": {
        "path": "app/tests/",
        "markers": "",
        "description": "Run all tests in sequence",
        "coverage_min": 80
    },
    "fast": {
        "path": "app/tests/unit/",
        "markers": "not performance",
        "description": "Run fast tests only (excluding performance)",
        "coverage_min": 80
    },
    "smoke": {
        "path": "app/tests/",
        "markers": "smoke",
        "description": "Run smoke tests for basic functionality",
        "coverage_min": 70
    }
}


class TestRunner:
    """Main test runner class with comprehensive functionality."""
    
    def __init__(self):
        self.start_time = time.time()
        self.results = []
        
    def run_test_suite(self, suite_name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific test suite and return results."""
        if suite_name not in TEST_CONFIGS:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        config = TEST_CONFIGS[suite_name]
        print(f"\n{'='*60}")
        print(f"Running {suite_name.upper()} Tests")
        print(f"Description: {config['description']}")
        print(f"Path: {config['path']}")
        print(f"{'='*60}\n")
        
        # Build pytest command
        cmd = self._build_pytest_command(config, **kwargs)
        
        # Execute tests
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        # Parse results
        test_result = {
            "suite": suite_name,
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "duration": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.results.append(test_result)
        
        # Print results
        self._print_test_results(test_result)
        
        return test_result
    
    def _build_pytest_command(self, config: Dict, **kwargs) -> List[str]:
        """Build pytest command with appropriate options."""
        cmd = ["python", "-m", "pytest"]
        
        # Add test path
        cmd.append(config["path"])
        
        # Add markers
        if config["markers"]:
            cmd.extend(["-m", config["markers"]])
        
        # Add coverage if requested
        if kwargs.get("coverage", True):
            cmd.extend([
                "--cov=app",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                f"--cov-fail-under={config['coverage_min']}"
            ])
        
        # Add verbosity
        verbose_level = kwargs.get("verbose", 1)
        if verbose_level == 1:
            cmd.append("-v")
        elif verbose_level >= 2:
            cmd.append("-vv")
        
        # Add specific options
        if kwargs.get("fail_fast", False):
            cmd.append("-x")
        
        if kwargs.get("last_failed", False):
            cmd.append("--lf")
        
        if kwargs.get("failed_first", False):
            cmd.append("--ff")
        
        # Add parallel execution for performance tests
        if "performance" in config.get("markers", ""):
            cmd.extend(["-n", "auto"])
        
        # Add output options
        cmd.extend([
            "--tb=short",
            "--color=yes",
            "--durations=10"
        ])
        
        return cmd
    
    def _print_test_results(self, result: Dict[str, Any]):
        """Print formatted test results."""
        status = "✅ PASSED" if result["passed"] else "❌ FAILED"
        duration = f"{result['duration']:.2f}s"
        
        print(f"\n{'-'*40}")
        print(f"Suite: {result['suite'].upper()}")
        print(f"Status: {status}")
        print(f"Duration: {duration}")
        print(f"{'-'*40}")
        
        if not result["passed"]:
            print("\nSTDOUT:")
            print(result["stdout"])
            print("\nSTDERR:")
            print(result["stderr"])
    
    def run_multiple_suites(self, suites: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Run multiple test suites in sequence."""
        results = []
        
        for suite in suites:
            try:
                result = self.run_test_suite(suite, **kwargs)
                results.append(result)
                
                # Stop on first failure if fail_fast is enabled
                if not result["passed"] and kwargs.get("fail_fast", False):
                    print(f"\n⚠️  Stopping test execution due to failure in {suite}")
                    break
                    
            except Exception as e:
                print(f"\n❌ Error running {suite}: {e}")
                results.append({
                    "suite": suite,
                    "passed": False,
                    "error": str(e),
                    "duration": 0
                })
        
        return results
    
    def print_summary(self):
        """Print overall test execution summary."""
        if not self.results:
            print("\nNo tests were executed.")
            return
        
        total_duration = time.time() - self.start_time
        passed_count = sum(1 for r in self.results if r["passed"])
        failed_count = len(self.results) - passed_count
        
        print(f"\n{'='*60}")
        print("TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Suites: {len(self.results)}")
        print(f"Passed: {passed_count} ✅")
        print(f"Failed: {failed_count} ❌")
        print(f"Success Rate: {(passed_count/len(self.results)*100):.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        if failed_count > 0:
            print("\nFailed Suites:")
            for result in self.results:
                if not result["passed"]:
                    print(f"  - {result['suite']} ({result.get('duration', 0):.2f}s)")
        
        print(f"{'='*60}\n")
        
        # Return exit code
        return 0 if failed_count == 0 else 1


def create_test_report(results: List[Dict[str, Any]], output_file: str = "test_report.html"):
    """Create HTML test report."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>fynqAI Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .suite { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .passed { background: #d4edda; border-color: #c3e6cb; }
        .failed { background: #f8d7da; border-color: #f5c6cb; }
        .summary { background: #e2e3e5; padding: 15px; border-radius: 5px; margin: 20px 0; }
        pre { background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>fynqAI Test Execution Report</h1>
        <p>Generated on: """ + time.strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Suites: """ + str(len(results)) + """</p>
        <p>Passed: """ + str(sum(1 for r in results if r["passed"])) + """</p>
        <p>Failed: """ + str(sum(1 for r in results if not r["passed"])) + """</p>
    </div>
"""
    
    for result in results:
        status_class = "passed" if result["passed"] else "failed"
        status_text = "PASSED" if result["passed"] else "FAILED"
        
        html_content += f"""
    <div class="suite {status_class}">
        <h3>{result['suite'].upper()} - {status_text}</h3>
        <p>Duration: {result.get('duration', 0):.2f}s</p>
        <p>Command: <code>{result.get('command', '')}</code></p>
        """
        
        if not result["passed"] and result.get("stdout"):
            html_content += f"<h4>Output:</h4><pre>{result['stdout']}</pre>"
        
        if not result["passed"] and result.get("stderr"):
            html_content += f"<h4>Errors:</h4><pre>{result['stderr']}</pre>"
        
        html_content += "</div>"
    
    html_content += """
</body>
</html>
"""
    
    with open(output_file, "w") as f:
        f.write(html_content)
    
    print(f"📊 Test report generated: {output_file}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="fynqAI Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites:
  unit         - Unit tests for individual components
  integration  - Integration tests for workflows
  performance  - Performance and load tests
  api          - API endpoint tests only
  ai           - AI core module tests only
  services     - Business service tests only
  workers      - Background worker tests only
  all          - All tests in sequence
  fast         - Fast tests only (no performance)
  smoke        - Smoke tests for basic functionality

Examples:
  python run_tests.py unit                    # Run unit tests
  python run_tests.py all --coverage          # Run all tests with coverage
  python run_tests.py fast --fail-fast        # Run fast tests, stop on first failure
  python run_tests.py performance --no-cov    # Run performance tests without coverage
        """
    )
    
    parser.add_argument(
        "suites",
        nargs="+",
        choices=list(TEST_CONFIGS.keys()),
        help="Test suite(s) to run"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable code coverage reporting"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first test failure"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (use -vv for extra verbose)"
    )
    
    parser.add_argument(
        "--last-failed",
        action="store_true",
        help="Run only tests that failed in the last run"
    )
    
    parser.add_argument(
        "--failed-first",
        action="store_true",
        help="Run failed tests first"
    )
    
    parser.add_argument(
        "--report",
        metavar="FILE",
        help="Generate HTML test report"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner()
    
    # Prepare kwargs
    kwargs = {
        "coverage": not args.no_coverage,
        "fail_fast": args.fail_fast,
        "verbose": args.verbose,
        "last_failed": args.last_failed,
        "failed_first": args.failed_first
    }
    
    try:
        # Run test suites
        if len(args.suites) == 1:
            runner.run_test_suite(args.suites[0], **kwargs)
        else:
            runner.run_multiple_suites(args.suites, **kwargs)
        
        # Generate report if requested
        if args.report:
            create_test_report(runner.results, args.report)
        
        # Print summary and exit
        exit_code = runner.print_summary()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
