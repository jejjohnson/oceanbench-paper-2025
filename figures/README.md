# OceanBench Figures


## Models

**"Truth"**
* [ ] Satellite, ARGO, etc (Observations)
* [ ] GLORYS - Reanalysis (Simulation + Observations)
* [ ] GLO12 - Analysis (Forecast + Observations)

**Models**
* Model (x3) (Forecast)
  * [ ] GLO12 (Physical)
  * [ ] GLONET (ML)
  * [ ] XiHe (ML)
  * [ ] WenHai (ML)


---
## Variables

* [ ] Temperature, $K$
* [ ] Salinity, $$
* [ ] Currents, [$m s^{-1}$]
* [ ] SSH, [$cm$]
* [ ] MLD, [$m$]
* [ ] Density, [$g cm^{-3}$]
* [ ] Geostrophic Currents, [$m s^{-1}$]
* [ ] Vorticity, [$s^{-1}$]
* [ ] Strain, 
* [ ] Lagrangian Trajectories, [$km$]

---
### Depth Levels

* [ ] Surface
* [ ] 50 m
* [ ] 200 m
* [ ] 300 m
* [ ] 600 m


---
## Region of Interest

* Gulfstream
* World
* Med.

---
## Events of Interest

* Hurricanes
* Typhoon
* Glacier Melting
* Collapse of the AMOC
* Marine HeatWaves

## Period of Interest

* 2024
* $t_0$ - Tuesday
* $t_1$ - +10 Days
* x52 Forecasts --> Average, STD, Quantiles

```python
# input data
U: Array[D S T] = ...
# rearrange to forecasts
U_F: Array[D S 10 F] = ...
# different models
U_FM: Array[D S 10 F M] = ...
```

---
## Lead Time


---
### Global Comparison Scores

**Metrics**

* RMSE
* MAE
* CRPS
* Relative RMSE
* Relative MAE
* Relative CRPS (?)


---
### Individual Analysis

* Maps
* Spectrum

---
