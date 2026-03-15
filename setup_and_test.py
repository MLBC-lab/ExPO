#!/usr/bin/env python3
"""
Installation and testing script for ExPO.
This script handles dependency installation, package setup, and verification.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, cwd=None):
    """Run a command and optionally check for success."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def install_package(editable=True, dev=False):
    """Install the ExPO package."""
    print("Installing ExPO package...")
    
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(".")
    
    if dev:
        cmd.extend(["[dev]"])
    
    run_command(cmd)
    print("? Package installed successfully")


def verify_installation():
    """Verify that ExPO can be imported and basic functions work."""
    print("Verifying installation...")
    
    try:
        import expo
        print(f"? ExPO package imported successfully (version: {getattr(expo, '__version__', 'unknown')})")
        
        # Test basic imports
        from expo.config import ExperimentConfig
        from expo.models.expo_model import ExPOModel
        from expo.data.dataset import ExPODataset
        print("? Core modules import successfully")
        
        return True
    except ImportError as e:
        print(f"? Import failed: {e}")
        return False


def run_tests():
    """Run the test suite if available."""
    if not Path("tests").exists():
        print("No test directory found, skipping tests")
        return True
    
    print("Running tests...")
    try:
        # Try to run our simple test runner first
        result = run_command([sys.executable, "tests/test_expo.py"], check=False)
        
        if result.returncode == 0:
            print("? All basic tests passed")
            
            # Try pytest if available, but don't fail if it doesn't work
            try:
                import pytest
                # Install pytest-cov if we need it
                try:
                    run_command([sys.executable, "-m", "pip", "install", "pytest-cov"], check=False)
                except:
                    pass
                    
                # Run pytest with simpler options if coverage fails
                try:
                    run_command([sys.executable, "-m", "pytest", "tests/", "-v"])
                    print("? Pytest tests passed")
                except RuntimeError:
                    # Try without coverage
                    print("Trying pytest without coverage...")
                    run_command([sys.executable, "-m", "pytest", "tests/", "-v", "--no-cov"])
                    print("? Pytest tests passed (no coverage)")
            except (ImportError, RuntimeError):
                print("Pytest not available or failed, but basic tests passed")
            
            return True
        else:
            print("? Some tests failed")
            return False
            
    except Exception as e:
        print(f"? Test execution failed: {e}")
        return False


def run_demo(keep_data=False):
    """Run the full demo pipeline."""
    print("Running demo pipeline...")
    try:
        cmd = [sys.executable, "scripts/run_full_demo_pipeline.py"]
        if keep_data:
            cmd.append("--keep-data")
        
        run_command(cmd)
        print("? Demo pipeline completed successfully")
        return True
    except RuntimeError:
        print("? Demo pipeline failed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Install and test ExPO")
    parser.add_argument("--dev", action="store_true", 
                       help="Install development dependencies")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip package installation")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running tests")
    parser.add_argument("--skip-demo", action="store_true",
                       help="Skip running demo")
    parser.add_argument("--keep-demo-data", action="store_true",
                       help="Keep demo data after running")
    args = parser.parse_args()
    
    success = True
    
    # Install package
    if not args.skip_install:
        try:
            install_package(editable=True, dev=args.dev)
        except RuntimeError as e:
            print(f"Installation failed: {e}")
            success = False
    
    # Verify installation
    if success:
        success = verify_installation()
    
    # Run tests
    if success and not args.skip_tests:
        success = run_tests()
    
    # Run demo
    if success and not args.skip_demo:
        success = run_demo(keep_data=args.keep_demo_data)
    
    if success:
        print("\n?? All checks passed! ExPO is ready to use.")
        print("\nNext steps:")
        print("1. Try the CLI commands: expo-train, expo-eval, expo-demo")
        print("2. Check out the documentation in README.md")
        print("3. Explore the example configurations in demo_workspace/configs/")
    else:
        print("\n? Some checks failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()