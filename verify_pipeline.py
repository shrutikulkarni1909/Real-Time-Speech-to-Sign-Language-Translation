#!/usr/bin/env python3
"""
PIPELINE VERIFICATION SCRIPT
=============================

This script verifies that all components of the speech-to-sign
translation pipeline are working correctly before running the
full evaluation.

Run this first to ensure everything is set up properly.
"""

import sys
import time

def check_imports():
    """Verify all required modules are installed"""
    print("Checking imports...")
    
    modules = [
        ("numpy", "Numerical computing"),
        ("sounddevice", "Microphone access"),
        ("whisper", "Speech transcription"),
        ("csv", "CSV file writing"),
        ("datetime", "Timestamp handling"),
        ("nltk", "Lemmatization/stemming"),
    ]
    
    all_ok = True
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"  ✓ {module_name:20} - {description}")
        except ImportError:
            print(f"  ✗ {module_name:20} - {description} [MISSING]")
            all_ok = False
    
    return all_ok


def check_backend_modules():
    """Verify backend pipeline modules"""
    print("\nChecking backend modules...")
    
    modules = [
        ("backend.gloss.infer", "Inference pipeline"),
        ("backend.gloss.asl_grammar", "ASL grammar preprocessing"),
        ("backend.gloss.word_resolver", "Word resolution"),
        ("backend.gloss.normalize", "Text normalization"),
        ("backend.gloss.dataset_loader", "Dictionary loading"),
    ]
    
    all_ok = True
    
    for module_path, description in modules:
        try:
            parts = module_path.split('.')
            module = __import__(module_path)
            for part in parts[1:]:
                module = getattr(module, part)
            print(f"  ✓ {module_path:35} - {description}")
        except (ImportError, AttributeError) as e:
            print(f"  ✗ {module_path:35} - {description} [ERROR]")
            print(f"    {e}")
            all_ok = False
    
    return all_ok


def check_dictionary():
    """Verify sign dictionary is loaded"""
    print("\nChecking sign dictionary...")
    
    try:
        from backend.gloss.infer import DAILY_GLOSS_MAP
        
        if DAILY_GLOSS_MAP:
            word_count = len(DAILY_GLOSS_MAP)
            print(f"  ✓ Dictionary loaded: {word_count} entries")
            
            # Show sample words
            sample_words = list(DAILY_GLOSS_MAP.keys())[:5]
            print(f"    Sample words: {', '.join(sample_words)}")
            
            return True
        else:
            print(f"  ✗ Dictionary is empty!")
            return False
    
    except Exception as e:
        print(f"  ✗ Error loading dictionary: {e}")
        return False


def check_pipeline_functions():
    """Verify pipeline functions are callable"""
    print("\nChecking pipeline functions...")
    
    functions = []
    
    try:
        from backend.gloss.infer import infer
        functions.append(("infer", infer))
    except ImportError:
        print(f"  ✗ Cannot import infer()")
        return False
    try:
        from backend.gloss.asl_grammar import convert_to_asl_order
        functions.append(("convert_to_asl_order", convert_to_asl_order))
    except ImportError:
        print(f"  ✗ Cannot import convert_to_asl_order()")
        return False
    try:
        from backend.gloss.word_resolver import resolve_word
        functions.append(("resolve_word", resolve_word))
    except ImportError:
        print(f"  ✗ Cannot import resolve_word()")
        return False
    
    # Test each function
    all_ok = True
    
    for name, func in functions:
        if callable(func):
            print(f"  ✓ {name:25} - Function is callable")
        else:
            print(f"  ✗ {name:25} - Not callable")
            all_ok = False
    
    return all_ok


def test_pipeline():
    """Test the pipeline with a sample input"""
    print("\nTesting pipeline with sample input...")
    
    try:
        from backend.gloss.infer import infer
        
        test_input = "I am eating pizza"
        print(f"  Input:  '{test_input}'")
        
        result = infer(test_input)
        print(f"  Output: '{result}'")
        
        if result and result != "(empty)":
            print(f"  ✓ Pipeline executed successfully")
            return True
        else:
            print(f"  ✗ Pipeline returned empty result")
            return False
    
    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_audio_devices():
    """Check if microphone is available"""
    print("\nChecking audio devices...")
    
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        if default_input is not None:
            device_info = sd.query_devices(default_input)
            print(f"  ✓ Default input device: {device_info['name']}")
            print(f"    Channels: {device_info['max_input_channels']}")
            return True
        else:
            print(f"  ✗ No default input device found")
            print(f"    Available devices:")
            for i, device in enumerate(devices):
                print(f"      {i}: {device['name']} (input: {device['max_input_channels']})")
            return False
    
    except Exception as e:
        print(f"  ✗ Error checking audio devices: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 80)
    print("SPEECH-TO-SIGN PIPELINE VERIFICATION")
    print("=" * 80)
    print()
    
    checks = [
        ("Required Modules", check_imports),
        ("Backend Modules", check_backend_modules),
        ("Sign Dictionary", check_dictionary),
        ("Pipeline Functions", check_pipeline_functions),
        ("Audio Devices", check_audio_devices),
        ("Pipeline Test", test_pipeline),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {check_name}: {e}")
            results[check_name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {check_name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ All checks passed! You're ready to run the evaluation.")
        print("\nRun the evaluation with:")
        print("    python evaluate_pipeline.py")
        return 0
    else:
        print(f"\n❌ {total - passed} checks failed. Fix the issues above before evaluating.")
        print("\nCommon fixes:")
        print("  1. Install missing modules: pip install numpy sounddevice openai-whisper")
        print("  2. Check that daily_pairs.txt exists: backend/gloss/daily_pairs.txt")
        print("  3. Verify microphone is connected: python -c \"import sounddevice; print(sounddevice.query_devices())\"")
        return 1


if __name__ == "__main__":
    sys.exit(main())
