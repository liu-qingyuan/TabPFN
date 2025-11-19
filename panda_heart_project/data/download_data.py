#!/usr/bin/env python3
"""
UCI Heart Disease Data Downloader
Downloads processed Cleveland, Hungarian, VA, and Switzerland datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UCI dataset URLs
UCI_DATASETS = {
    'Cleveland': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'filename': 'cleveland.csv',
        'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    },
    'Hungarian': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
        'filename': 'hungarian.csv',
        'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    },
    'VA': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data',
        'filename': 'va.csv',
        'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    },
    'Switzerland': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
        'filename': 'switzerland.csv',
        'columns': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    }
}

# Clinical feature descriptions
CLINICAL_FEATURES = {
    'age': 'Age (continuous)',
    'sex': 'Sex (1=male, 0=female)',
    'cp': 'Chest pain type (1-4)',
    'trestbps': 'Resting blood pressure (mmHg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
    'restecg': 'Resting ECG (0-2)',
    'thalach': 'Maximum heart rate achieved (bpm)',
    'exang': 'Exercise induced angina (1=yes, 0=no)',
    'oldpeak': 'ST depression induced by exercise (continuous)',
    'slope': 'Slope of peak exercise ST segment (1-3)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)',
    'num': 'Diagnosis (0=no heart disease, 1-4=heart disease)'
}

def download_dataset(url: str, filepath: Path) -> bool:
    """Download dataset from UCI repository"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(response.text)

        logger.info(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def load_and_process_dataset(filepath: Path, columns: List[str], center_name: str) -> pd.DataFrame:
    """Load and process individual dataset"""
    try:
        # Read data - UCI datasets use ? for missing values
        df = pd.read_csv(filepath, names=columns, na_values='?')

        # Add center identifier
        df['center'] = center_name

        # Convert target to binary (0 vs 1-4)
        df['target'] = (df['num'] > 0).astype(int)

        # Basic data cleaning
        df = df.dropna(subset=['age', 'sex'])  # Must have age and sex

        logger.info(f"Loaded {center_name}: {len(df)} samples, {df.isnull().sum().sum()} missing values")
        return df

    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        return pd.DataFrame()

def analyze_missing_patterns(df: pd.DataFrame, center_name: str) -> Dict:
    """Analyze missing data patterns for each center"""
    missing_stats = {}
    total_samples = len(df)

    for col in df.columns:
        if col in ['center', 'target', 'num']:
            continue

        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_samples) * 100
        missing_stats[col] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2)
        }

    overall_missing = df.isnull().sum().sum()
    logger.info(f"{center_name} missing analysis: {overall_missing} total missing values "
               f"({(overall_missing/(total_samples*14))*100:.2f}% missing rate)")

    return missing_stats

def main():
    """Main download and processing function"""
    logger.info("🚀 Starting UCI Heart Disease Dataset Download")

    # Create data directory
    data_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = []
    missing_analysis = {}

    # Download and process each dataset
    for center_name, config in UCI_DATASETS.items():
        logger.info(f"\n📊 Processing {center_name} dataset...")

        # Download raw data
        raw_filepath = data_dir / config['filename']
        if not download_dataset(config['url'], raw_filepath):
            continue

        # Process dataset
        df = load_and_process_dataset(raw_filepath, config['columns'], center_name)
        if not df.empty:
            all_datasets.append(df)

            # Analyze missing patterns
            missing_stats = analyze_missing_patterns(df, center_name)
            missing_analysis[center_name] = missing_stats

    if not all_datasets:
        logger.error("❌ No datasets were successfully processed")
        return

    # Combine all datasets
    logger.info("\n🔗 Combining all datasets...")
    combined_df = pd.concat(all_datasets, ignore_index=True)

    # Save combined dataset
    combined_filepath = processed_dir / "uci_heart_disease_combined.csv"
    combined_df.to_csv(combined_filepath, index=False)

    # Save missing analysis
    missing_summary_filepath = processed_dir / "missing_analysis_summary.json"
    import json
    with open(missing_summary_filepath, 'w') as f:
        json.dump(missing_analysis, f, indent=2)

    # Print summary statistics
    logger.info(f"\n✅ Dataset processing completed!")
    logger.info(f"📈 Combined dataset: {len(combined_df)} samples from {len(all_datasets)} centers")

    # Center breakdown
    center_counts = combined_df['center'].value_counts()
    logger.info("\n📊 Samples per center:")
    for center, count in center_counts.items():
        logger.info(f"  {center}: {count} samples")

    # Target distribution
    target_counts = combined_df['target'].value_counts()
    logger.info(f"\n🎯 Target distribution:")
    logger.info(f"  No disease (0): {target_counts.get(0, 0)} samples")
    logger.info(f"  Heart disease (1): {target_counts.get(1, 0)} samples")

    # Overall missing rate
    overall_missing = combined_df.isnull().sum().sum()
    total_cells = len(combined_df) * len([col for col in combined_df.columns if col not in ['center', 'target', 'num']])
    missing_rate = (overall_missing / total_cells) * 100
    logger.info(f"\n❓ Overall missing rate: {missing_rate:.2f}%")

    logger.info(f"\n💾 Files saved:")
    logger.info(f"  Raw data: {data_dir}")
    logger.info(f"  Combined dataset: {combined_filepath}")
    logger.info(f"  Missing analysis: {missing_summary_filepath}")

if __name__ == "__main__":
    main()