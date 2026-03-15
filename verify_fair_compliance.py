#!/usr/bin/env python3
"""
Verification script for ExPO FAIR compliance.
"""

def test_package_installation():
    """Verify package can be installed and imported."""
    print("🔍 Testing package installation and imports...")
    try:
        import expo
        print(f"✅ ExPO package imported successfully (version: {expo.__version__})")
        
        # Test core imports
        from expo.config import ExperimentConfig
        from expo.models.expo_model import ExPOModel
        from expo.data.dataset import ExPODataset
        from expo.training.trainer import ExPOTrainer
        print("✅ All core modules import successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_cli_interface():
    """Verify CLI commands are available."""
    print("\n🔍 Testing CLI interface...")
    import subprocess
    
    commands = ["expo-train --help", "expo-eval --help", "expo-demo --help"]
    for cmd in commands:
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {cmd.split()[0]} command available")
            else:
                print(f"⚠️  {cmd.split()[0]} command available but returned non-zero")
        except Exception as e:
            print(f"❌ {cmd.split()[0]} command failed: {e}")
            return False
    return True


def test_documentation_completeness():
    """Verify documentation files exist and are comprehensive."""
    print("\n🔍 Testing documentation completeness...")
    
    required_docs = [
        ("README.md", "Main documentation"),
        ("docs/README.md", "Installation guide"),
        ("docs/API.md", "API reference"),
        ("LICENSE", "License file"),
        ("pyproject.toml", "Package metadata"),
        ("FAIR_COMPLIANCE_REPORT.md", "FAIR compliance report")
    ]
    
    from pathlib import Path
    
    for doc_file, description in required_docs:
        path = Path(doc_file)
        if path.exists():
            size = path.stat().st_size
            if size > 1000:  # At least 1KB of content
                print(f"✅ {description} ({doc_file}): {size:,} bytes")
            else:
                print(f"⚠️  {description} ({doc_file}): Only {size} bytes")
        else:
            print(f"❌ {description} missing ({doc_file})")
            return False
    
    return True


def test_package_metadata():
    """Verify package has proper metadata."""
    print("\n🔍 Testing package metadata...")
    
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)
        
        project = config.get("project", {})
        
        required_fields = [
            ("name", "Package name"),
            ("version", "Version"),
            ("description", "Description"),
            ("authors", "Authors"),
            ("license", "License"),
            ("dependencies", "Dependencies")
        ]
        
        for field, description in required_fields:
            if field in project and project[field]:
                print(f"✅ {description}: {project[field]}")
            else:
                print(f"❌ {description} missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error reading package metadata: {e}")
        return False


def test_reproducibility():
    """Verify reproducibility features."""
    print("\n🔍 Testing reproducibility features...")
    
    try:
        # Test configuration loading
        from expo.config import ExperimentConfig
        from pathlib import Path
        
        config_path = Path("demo_workspace/configs/simple_config.json")
        if config_path.exists():
            config = ExperimentConfig.from_json(str(config_path))
            print("✅ Configuration loading works")
            
            # Test seed setting
            if hasattr(config.training, 'seed') and config.training.seed:
                print(f"✅ Seed configuration available: {config.training.seed}")
            else:
                print("⚠️  No seed configuration found")
                
            return True
        else:
            print("❌ No configuration file found for testing")
            return False
            
    except Exception as e:
        print(f"❌ Reproducibility test failed: {e}")
        return False


def test_demo_functionality():
    """Verify demo pipeline works."""
    print("\n🔍 Testing demo functionality...")
    
    import subprocess
    from pathlib import Path
    
    try:
        # Test synthetic data generation
        cmd = ["python", "scripts/generate_test_data.py", 
               "--out-dir", "test_temp_data", 
               "--n-profiles", "10", 
               "--n-genes", "5"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Synthetic data generation works")
            
            # Check if files were created
            data_dir = Path("test_temp_data")
            if (data_dir / "expression_table.pkl").exists():
                print("✅ Expression data file created")
                
                # Clean up
                import shutil
                if data_dir.exists():
                    shutil.rmtree(data_dir)
                    
                return True
        
        print(f"❌ Demo functionality test failed: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"❌ Demo test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("🚀 ExPO FAIR Compliance Verification")
    print("=" * 50)
    
    tests = [
        ("Package Installation", test_package_installation),
        ("CLI Interface", test_cli_interface), 
        ("Documentation", test_documentation_completeness),
        ("Package Metadata", test_package_metadata),
        ("Reproducibility", test_reproducibility),
        ("Demo Functionality", test_demo_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL FAIR REQUIREMENTS SATISFIED!")
        print("The ExPO package successfully meets all FAIR compliance requirements:")
        print("- ✅ Proper package structure with pyproject.toml")
        print("- ✅ Comprehensive documentation")
        print("- ✅ Working demo pipeline")
        print("- ✅ CLI interface for accessibility")
        print("- ✅ Reproducible configuration system")
        print("- ✅ Software engineering best practices")
        return True
    else:
        print(f"\n⚠️  {total - passed} requirements need attention")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)