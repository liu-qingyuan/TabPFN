"""
Quick test of adapt library implementations for heart disease data
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

# Test adapt library
try:
    from adapt.feature_based import CORAL, TCA
    print("✅ adapt library imported successfully")
except ImportError as e:
    print(f"❌ adapt library import failed: {e}")
    exit(1)

# Test our adapt adapter
try:
    from models.adapt_domain_adapter import AdaptDomainAdapter
    print("✅ AdaptDomainAdapter imported successfully")
except ImportError as e:
    print(f"❌ AdaptDomainAdapter import failed: {e}")
    exit(1)

# Test data loader
try:
    from scripts.data.heart_disease_loader import HeartDiseaseDataLoader
    print("✅ HeartDiseaseDataLoader imported successfully")
except ImportError as e:
    print(f"❌ HeartDiseaseDataLoader import failed: {e}")
    exit(1)

def test_adapt_methods():
    """Test adapt library domain adaptation methods"""
    print("\n🔍 Testing adapt library methods...")

    # Load data
    loader = HeartDiseaseDataLoader()
    loader.load_data()

    # Get sample data
    X_source, y_source, X_target, y_target = loader.create_domain_adaptation_split('Cleveland', 'Hungarian')

    # Convert to numpy
    if hasattr(X_source, 'values'):
        X_source = X_source.values
    if hasattr(X_target, 'values'):
        X_target = X_target.values
    if hasattr(y_source, 'values'):
        y_source = y_source.values
    if hasattr(y_target, 'values'):
        y_target = y_target.values

    print(f"Data shapes: Source={X_source.shape}, Target={X_target.shape}")
    print(f"Unique labels: Source={np.unique(y_source)}, Target={np.unique(y_target)}")

    # Test different adapt methods
    methods = {
        'CORAL': CORAL(),  # Use default parameters
        'TCA': TCA(kernel='linear', mu=1.0, Xt=X_target)
    }

    for method_name, method in methods.items():
        try:
            print(f"\n📊 Testing {method_name}:")

            # Fit and transform
            if method_name == 'TCA':
                method.fit(X_source, y_source)
                X_source_transformed = method.transform(X_source)
                X_target_transformed = method.transform(X_target)
            else:  # CORAL
                method.fit(X_source, X_target)
                X_source_transformed = method.transform(X_source)
                X_target_transformed = method.transform(X_target)

            print(f"  Original shapes: Source={X_source.shape}, Target={X_target.shape}")
            print(f"  Transformed shapes: Source={X_source_transformed.shape}, Target={X_target_transformed.shape}")

            # Calculate domain distance
            source_mean = np.mean(X_source_transformed, axis=0)
            target_mean = np.mean(X_target_transformed, axis=0)
            distance = np.linalg.norm(source_mean - target_mean)
            print(f"  Domain distance after adaptation: {distance:.4f}")

            # Calculate original distance for comparison
            orig_source_mean = np.mean(X_source, axis=0)
            orig_target_mean = np.mean(X_target, axis=0)
            orig_distance = np.linalg.norm(orig_source_mean - orig_target_mean)
            print(f"  Original domain distance: {orig_distance:.4f}")
            print(f"  Distance reduction: {(orig_distance - distance) / orig_distance * 100:.1f}%")

            print(f"  ✅ {method_name} test successful!")

        except Exception as e:
            print(f"  ❌ {method_name} test failed: {e}")

def test_adapt_domain_adapter():
    """Test our AdaptDomainAdapter wrapper"""
    print("\n🔧 Testing AdaptDomainAdapter...")

    try:
        # Load data
        loader = HeartDiseaseDataLoader()
        loader.load_data()
        X_source, y_source, X_target, y_target = loader.create_domain_adaptation_split('Cleveland', 'Hungarian')

        # Test CORAL adapter
        print("\n📊 Testing AdaptDomainAdapter with CORAL:")
        coral_adapter = AdaptDomainAdapter(adaptation_method='CORAL')
        coral_adapter.fit(X_source, y_source, X_target)

        # Make predictions
        y_pred = coral_adapter.predict(X_target, domain='target')
        y_pred_proba = coral_adapter.predict_proba(X_target, domain='target')

        print(f"  Predictions shape: {y_pred.shape}")
        print(f"  Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"  Unique predictions: {np.unique(y_pred)}")
        print(f"  Prediction accuracy: {np.mean(y_pred == y_target):.3f}")
        print(f"  ✅ CORAL adapter test successful!")

        # Test TCA adapter
        print("\n📊 Testing AdaptDomainAdapter with TCA:")
        tca_adapter = AdaptDomainAdapter(adaptation_method='TCA')
        tca_adapter.fit(X_source, y_source, X_target)

        # Make predictions
        y_pred = tca_adapter.predict(X_target, domain='target')
        y_pred_proba = tca_adapter.predict_proba(X_target, domain='target')

        print(f"  Predictions shape: {y_pred.shape}")
        print(f"  Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"  Unique predictions: {np.unique(y_pred)}")
        print(f"  Prediction accuracy: {np.mean(y_pred == y_target):.3f}")
        print(f"  ✅ TCA adapter test successful!")

    except Exception as e:
        print(f"  ❌ AdaptDomainAdapter test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🧪 PANDA-Heart Adapt Library Test")
    print("="*50)

    test_adapt_methods()
    test_adapt_domain_adapter()

    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()