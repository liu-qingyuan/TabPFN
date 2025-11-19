"""
Test using adapt library's CORAL vs our custom implementation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Test adapt library availability
try:
    from adapt.feature_based import CORAL as AdaptCORAL
    ADAPT_AVAILABLE = True
    print("✅ adapt library is available")
except ImportError:
    ADAPT_AVAILABLE = False
    print("❌ adapt library not available")

# Import our custom implementations
from models.panda_heart_adapter import CORALMethod, TCAMethod
from scripts.data.heart_disease_loader import HeartDiseaseDataLoader

def compare_implementations():
    """Compare custom vs adapt library implementations"""

    if not ADAPT_AVAILABLE:
        print("Cannot compare - adapt library not installed")
        return

    print("🔍 Comparing Custom vs Adapt Library Implementations")
    print("="*60)

    # Load data
    loader = HeartDiseaseDataLoader()
    loader.load_data()
    X_source, y_source, X_target, y_target = loader.create_domain_adaptation_split('Cleveland', 'Hungarian')

    # Convert to numpy arrays
    X_source_np = X_source.values if hasattr(X_source, 'values') else X_source
    X_target_np = X_target.values if hasattr(X_target, 'values') else X_target

    print(f"Data shapes: Source={X_source_np.shape}, Target={X_target_np.shape}")

    # Test implementations
    methods = {
        'Custom CORAL': CORALMethod(reg_param=1e-6),
        'Adapt CORAL': AdaptCORAL()
    }

    results = {}

    for method_name, method in methods.items():
        try:
            print(f"\n📊 Testing {method_name}:")

            if method_name == 'Adapt CORAL':
                # Adapt library interface
                method.fit(X_source_np, X_target_np)
                X_src_transformed = method.transform(X_source_np)
                X_tgt_transformed = method.transform(X_target_np)
            else:
                # Custom interface
                X_src_transformed, X_tgt_transformed = method.fit_transform(
                    X_source_np, X_target_np, y_source
                )

            # Calculate domain distance
            source_mean = np.mean(X_src_transformed, axis=0)
            target_mean = np.mean(X_tgt_transformed, axis=0)
            distance = np.linalg.norm(source_mean - target_mean)

            results[method_name] = {
                'source_shape': X_src_transformed.shape,
                'target_shape': X_tgt_transformed.shape,
                'domain_distance': distance,
                'success': True
            }

            print(f"  ✅ Success! Shapes: {X_src_transformed.shape}, Distance: {distance:.4f}")

        except Exception as e:
            results[method_name] = {'error': str(e), 'success': False}
            print(f"  ❌ Error: {e}")

    # Comparison
    successful = {k: v for k, v in results.items() if v['success']}
    if len(successful) == 2:
        custom_dist = successful['Custom CORAL']['domain_distance']
        adapt_dist = successful['Adapt CORAL']['domain_distance']

        print(f"\n🏆 Comparison Results:")
        print(f"  Custom CORAL domain distance: {custom_dist:.4f}")
        print(f"  Adapt CORAL domain distance: {adapt_dist:.4f}")
        print(f"  Difference: {abs(custom_dist - adapt_dist):.4f}")

        if adapt_dist < custom_dist:
            print("  ✅ Adapt CORAL performs better (smaller domain distance)")
        else:
            print("  🤔 Custom CORAL performs better or equal")

if __name__ == "__main__":
    compare_implementations()