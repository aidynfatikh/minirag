#!/usr/bin/env python3
"""
Test file structure and configuration loading (no heavy dependencies)
"""

import sys
import json
from pathlib import Path

# Test that basic file structure exists
def test_file_structure():
    """Test that required files exist"""
    print("="*80)
    print("Testing File Structure")
    print("="*80)
    
    base_path = Path(__file__).parent.parent
    
    required_files = {
        'config/config.yaml': 'Configuration file',
        'config/test_cases.json': 'Test cases',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Documentation',
        'minirag/__init__.py': 'Package init',
        'minirag/rag.py': 'Core RAG module',
        'minirag/generator.py': 'Generator module',
        'minirag/pdf_parser.py': 'PDF parser',
        'minirag/config_loader.py': 'Config loader',
        'scripts/index.py': 'Indexing script',
        'scripts/query.py': 'Query script',
        'scripts/evaluate.py': 'Evaluation script',
    }
    
    missing = []
    for filename, description in required_files.items():
        filepath = base_path / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing) - {description}")
            missing.append(filename)
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} files")
        return False
    else:
        print("\n✓ All required files present!")
        return True


def test_config_yaml():
    """Test that config.yaml is valid YAML"""
    print("\n" + "="*80)
    print("Testing config.yaml")
    print("="*80)
    
    try:
        import yaml
        
        base_path = Path(__file__).parent.parent
        config_path = base_path / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check main sections
        required_sections = ['models', 'indexing', 'search', 'generation', 'paths']
        for section in required_sections:
            if section in config:
                print(f"  ✓ {section} section present")
            else:
                print(f"  ✗ {section} section missing")
                return False
        
        print("\n✓ Config YAML is valid and has all required sections!")
        return True
        
    except ImportError:
        print("  ⚠ PyYAML not installed, skipping YAML validation")
        return True
    except Exception as e:
        print(f"  ✗ Error loading config: {e}")
        return False


def test_test_cases_json():
    """Test that test_cases.json is valid JSON"""
    print("\n" + "="*80)
    print("Testing test_cases.json")
    print("="*80)
    
    try:
        base_path = Path(__file__).parent.parent
        test_cases_path = base_path / "config" / "test_cases.json"
        
        with open(test_cases_path, 'r') as f:
            test_cases = json.load(f)
        
        if not isinstance(test_cases, list):
            print(f"  ✗ Test cases should be a list, got {type(test_cases)}")
            return False
        
        print(f"  ✓ Loaded {len(test_cases)} test cases")
        
        # Check first test case structure
        if test_cases:
            first_case = test_cases[0]
            required_fields = ['id', 'query', 'expected_pages']
            for field in required_fields:
                if field in first_case:
                    print(f"  ✓ Test cases have '{field}' field")
                else:
                    print(f"  ✗ Test cases missing '{field}' field")
                    return False
        
        print("\n✓ Test cases JSON is valid!")
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading test cases: {e}")
        return False


def test_imports():
    """Test that package structure allows imports"""
    print("\n" + "="*80)
    print("Testing Package Imports")
    print("="*80)
    
    base_path = Path(__file__).parent.parent
    sys.path.insert(0, str(base_path))
    
    try:
        # Test config_loader imports (doesn't need heavy dependencies)
        print("  Attempting to import config module...")
        from minirag import config_loader
        print("  ✓ Config loader imported successfully")
        
        # Check that Config class exists
        if hasattr(config_loader, 'Config'):
            print("  ✓ Config class found")
        else:
            print("  ✗ Config class not found")
            return False
        
        # Check that get_config function exists
        if hasattr(config_loader, 'get_config'):
            print("  ✓ get_config function found")
        else:
            print("  ✗ get_config function not found")
            return False
        
        print("\n✓ Package imports work correctly!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        print("\n  Note: Some dependencies may not be installed yet.")
        print("  Run: pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MiniRAG Structure Tests (No Dependencies Required)")
    print("="*80)
    
    results = []
    results.append(("File Structure", test_file_structure()))
    results.append(("Config YAML", test_config_yaml()))
    results.append(("Test Cases JSON", test_test_cases_json()))
    results.append(("Package Imports", test_imports()))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All structure tests passed!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run full tests: python3 tests/test_config.py")
        print("  3. Add PDFs to pdfs/ directory")
        print("  4. Run: python3 scripts/index.py")
        print("  5. Run: python3 scripts/query.py")
    else:
        print("\n⚠ Some tests failed. Please check the output above.")
    
    print("="*80)
