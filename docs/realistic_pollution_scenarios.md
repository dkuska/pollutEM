# Realistic Pollution Scenarios

## 2MASS

ra (Right Ascension):
    Rounding: Reduce decimal precision from 6 to 2-3 places
    Unit Conversion: Convert from decimal degrees to hours/minutes/seconds
    Gaussian Noise: Add small perturbations (~0.0001-0.001°)

decl (Declination):
    Rounding: Reduce decimal precision from 6 to 2-3 places
    Gaussian Noise: Add small perturbations (~0.0001-0.001°)
    Unit Conversion: Convert from decimal degrees to degrees/arcminutes/arcseconds

jdate (Julian Date):
    Rounding: Round to nearest day instead of fraction
    Shifting: Add/subtract small time offsets (1-2 days)
    Unit Conversion: Convert to Modified Julian Date

j_m, h_m, k_m (Magnitude measurements):
    Rounding: Round to nearest 0.1 or 0.5
    Gaussian Noise: Add noise with sigma ≈ 0.05-0.1
    Scaling: Multiply by small factor (0.95-1.05)

j_msigcom, h_msigcom, k_msigcom (Uncertainties):
    Rounding: Round to nearest 0.05
    Scaling: Multiply by factor (1.5-2.0)
    Null Value Insertion

## Earthquakes

depth:
    Unit Conversion (meters to feet)
    Rounding (to nearest 100m)
    Shifting (±500m systematic error)

depth_uncertainty:
    Rounding (to nearest 100)
    Addition of Gaussian Noise (sigma = 10% of value)

horizontal_uncertainty:
    Scaling (multiply by 1.1)
    Addition of Gaussian Noise (sigma = 5% of value)

used_phase_count, used_station_count:
    Null Value Insertion
    Swapping of Digits

standard_error:
    Rounding (to nearest 0.1)
    Scaling (multiply by 1.05)

mag_value:
    Rounding (to nearest 0.5)
    Addition of Gaussian Noise (sigma = 0.1)
    Scaling (multiply by 1.02)

mag_uncertainty:
    Scaling (x10)
    Null Value Insertion

## Protein

Volume/Size Metrics:
    volume: {Scaling (±10%), Gaussian noise (sigma=1.0), Unit conversion (Å³ to nm³)}
    surface: {Rounding (1 decimal), Scaling (±5%), Unit conversion (Å² to nm²)}
    depth: {Rounding (nearest integer), Shifting (±0.5), Gaussian noise (sigma=0.2)}

Ratios/Normalized Values (0-1 range):
    surf/vol: {Rounding (3 decimals), Scaling (±2%), Digit swapping}
    lid/hull: {Rounding (4 decimals), Gaussian noise (sigma=0.01)}
    ell_c/a, ell_b/a: {Rounding (3 decimals), Scaling (±1%)}

Count Data (integers):
    surfGPs, lidGPs, hullGPs: {Null insertion, Shifting (±1)}
    siteAtms: {Rounding (nearest 5), Null insertion}

Amino Acid Compositions (fractional):
    acidicAA, basicAA, etc.: {Rounding (2 decimals), Scaling (±2%), Null insertion}
    Individual AAs (ALA, ARG, etc.): {Rounding (3 decimals), Gaussian noise (sigma=0.05)}

## VSX

RAdeg/DEdeg (coordinates):
    Rounding (common in astronomical measurements)
    Addition of Gaussian Noise (measurement errors)

max/min (magnitude values):
    Rounding
    Scaling (confusion in magnitude systems)

Period:
    Rounding
    Unit Conversion (days vs hours confusion)
