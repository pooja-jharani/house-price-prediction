# Model Card: House Price Regressor

## Overview
- **Task:** predict house sale price (INR) from six features.
- **Model:** Random Forest regressor (tuned with randomized search) inside an `sklearn` pipeline that includes preprocessing.
- **Version:** 1.0.0. Training metadata (timestamp, library versions, metrics) is written to `models/metadata.json` on every run.

## Intended use
Learning, portfolio demonstration and as a template for a tabular regression workflow. **Not** intended for pricing, lending, insurance or any real financial decision.

## Training data
1,200 rows of synthetic data from `house_price.data.generate_synthetic_data` (seed 42). Features: `area_sqft`, `bedrooms`, `bathrooms`, `age_years`, `distance_to_city_km`, `location`. About 3% of `area_sqft`, `bathrooms` and `age_years` are missing and imputed in the pipeline. Split: 80% train / 20% test (`random_state=42`).

## Evaluation (held-out test set, n=240)
| Metric | Value |
|---|---|
| R² | 0.960 |
| RMSE | ₹519,215 |
| MAE | ₹276,067 |
| MAPE | 5.1% |

## Behaviour and known weaknesses
- Area and location account for nearly all predictive signal.
- Random Forests cannot extrapolate: houses far outside the training range (area < 400 or > 4,500 sq ft, age > 80 years, distance > 50 km) will be predicted poorly. The inference API accepts a wider range than the training data to avoid blocking legitimate queries, so treat extremes with caution.
- Unseen locations are silently encoded as all-zeros and are not meaningfully priced.
- Error is larger in absolute terms for expensive houses (see `reports/figures/residuals.png`).

## Ethical considerations
Location is the second most influential feature. On real data, location can proxy for protected attributes, so a fairness review would be needed before any real-world use.

## Reproducing
```bash
house-price generate-data && house-price train --tune
```
