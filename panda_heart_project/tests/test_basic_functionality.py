#!/usr/bin/env python3
"""
Basic functionality test for PANDA-Heart project
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules directly
import importlib.util
loader_spec = importlib.util.spec_from_file_location("heart_disease_loader", project_root / "scripts" / "data" / "heart_disease_loader.py")
loader_module = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(loader_module)
load_heart_disease_data = loader_module.load_heart_disease_data

baseline_spec = importlib.util.spec_from_file_location("baseline_models", project_root / "models" / "baseline_models.py")
baseline_module = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(baseline_module)
BaselineModelFactory = baseline_module.BaselineModelFactory

def test_data_loading():
    """Test basic data loading functionality"""
    print("🔍 Testing data loading...")

    try:
        loader = load_heart_disease_data()
        print("✅ Data loader created successfully")

        # Check what centers are available
        print(f"✅ Full dataset loaded: {len(loader.df)} samples")
        print(f"✅ Available centers: {loader.df['center'].unique()}")

        # Test getting center data with correct case
        available_centers = loader.df['center'].unique()
        for center in available_centers:
            center_data = loader.get_center_data(center)
            print(f"✅ {center} data loaded: {len(center_data)} samples")

        # Test feature preprocessing
        if len(available_centers) > 0:
            center_data = loader.get_center_data(available_centers[0])
            feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

            X = center_data[feature_names].copy()
            y = center_data['target'].copy()

            X_processed = loader.preprocess_features(X, fit_scalers=True, center_name="test")
            print(f"✅ Feature preprocessing successful: {X_processed.shape}")

        return True

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_baseline_models():
    """Test baseline model creation"""
    print("\n🔍 Testing baseline models...")

    try:
        factory = BaselineModelFactory()

        # Test creating a simple model
        model = factory.create_model('LASSO_LR', random_state=42)
        print("✅ LASSO_LR model created successfully")

        # Test data for model fitting
        X = np.random.randn(100, 13)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        print(f"✅ Model fitting successful: predictions shape {predictions.shape}")

        return True

    except Exception as e:
        print(f"❌ Baseline model test failed: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🚀 PANDA-Heart Basic Functionality Test")
    print("=" * 50)

    tests = [
        test_data_loading,
        test_baseline_models
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed! Ready for experiments.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before running experiments.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)