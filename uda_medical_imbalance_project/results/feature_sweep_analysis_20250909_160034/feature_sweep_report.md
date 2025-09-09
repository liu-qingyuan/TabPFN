# Feature Sweep Analysis Report

Generated: 2025-09-09 16:27:27

## Analysis Configuration

- Feature Range: best3 ~ best58
- Parallel Workers: 4
- Total Feature Sets: 13
- Successful Analyses: 13
- Failed Analyses: 0

## Performance Summary

### Best Performance Results

- **Best Source Domain**: best11 (AUC: 0.8444)
- **Best Target Baseline**: best3 (AUC: 0.6980)
- **Best Target TCA**: best3 (AUC: 0.7046)
- **Best TCA Improvement**: best10 (+0.0207)

### Performance Trends

- Average Source AUC: 0.8221
- Average Target Baseline AUC: 0.6933
- Average Target TCA AUC: 0.6999
- Average TCA Improvement: 0.0066

### Detailed Results

| Feature Set | N Features | Source AUC | Target Baseline | Target TCA | TCA Improvement |
|-------------|------------|------------|-----------------|------------|-----------------|
| best3 | 3 | 0.7709 | 0.6980 | 0.7046 | +0.0066 |
| best4 | 4 | 0.7661 | 0.6980 | 0.7046 | +0.0066 |
| best5 | 5 | 0.7988 | 0.6980 | 0.7046 | +0.0066 |
| best6 | 6 | 0.8153 | 0.6980 | 0.7046 | +0.0066 |
| best7 | 7 | 0.8280 | 0.6815 | 0.6916 | +0.0101 |
| best8 | 8 | 0.8322 | 0.6980 | 0.7046 | +0.0066 |
| best9 | 9 | 0.8405 | 0.6926 | 0.6812 | -0.0113 |
| best10 | 10 | 0.8429 | 0.6594 | 0.6801 | +0.0207 |
| best11 | 11 | 0.8444 | 0.6980 | 0.7046 | +0.0066 |
| best12 | 12 | 0.8381 | 0.6980 | 0.7046 | +0.0066 |
| best15 | 15 | 0.8437 | 0.6980 | 0.7046 | +0.0066 |
| best20 | 20 | 0.8414 | 0.6980 | 0.7046 | +0.0066 |
| best32 | 32 | 0.8251 | 0.6980 | 0.7046 | +0.0066 |

## Recommendations

- **Recommended Feature Set**: best11
  - Source Performance: 0.8444 AUC
  - Target Performance: 0.7046 AUC
  - TCA Improvement: +0.0066
- **TCA Domain Adaptation** shows positive effects for most feature sets

Detailed results and visualizations available in: results/feature_sweep_analysis_20250909_160034