import os
import pandas as pd
import numpy as np
from tableshift import get_dataset
from tableshift.core.features import PreprocessorConfig

def inspect_dataset(name, cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.abspath('data/raw')
    print(f"=== Inspecting {name} ===")
    print(f"Cache dir: {cache_dir}")
    try:
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load dataset
        # Note: TableShift datasets usually download automatically to cache_dir
        # Set use_cached=False to ensure it downloads/processes raw data first if not present
        dset = get_dataset(name, cache_dir=cache_dir, initialize_data=True, use_cached=False)
        
        # Access the source (train) and target (test) splits
        try:
            X_train, y_train, _, _ = dset.get_pandas('train')
            print(f"Train shape: {X_train.shape}, Positive rate: {y_train.mean():.4f}")
        except Exception as e:
            print(f"Could not load train split: {e}")

        try:
            # TableShift uses 'ood_test' for the target OOD test set
            target_split = 'ood_test' 
            X_test, y_test, _, _ = dset.get_pandas(target_split)
            print(f"Test ({target_split}) shape: {X_test.shape}, Positive rate: {y_test.mean():.4f}")
        except Exception as e:
             print(f"Could not load {target_split} split: {e}")
        
        # Check for shift identifier
        # For diabetes (BRFSS), it's usually 'race'
        # For hospital readmission, it's 'admission_source_id' (or similar)
        
        print("Train columns:", X_train.columns.tolist()[:5], "...")
        
        # Save a small sample to csv for quick look if needed
        X_train.head().to_csv(f"data/{name}_train_sample.csv")
        
        return dset
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

if __name__ == "__main__":
    # 1. Diabetes (BRFSS) - Race Shift
    inspect_dataset('diabetes_readmission')
    
    # 2. Hospital Readmission - Institution Shift
    # Note: In TableShift, the name for readmission might be 'diabetes_readmission' 
    # and the BRFSS one is 'brfss_diabetes' or 'diabetes'.
    # Let's check the documentation mapping or try common names.
    # Based on TableShift papers:
    # - 'diabetes_readmission' is the UCI Hospital Readmission dataset
    # - 'brfss_diabetes' is the BRFSS Diabetes dataset (Race shift)
    
    inspect_dataset('brfss_diabetes')
