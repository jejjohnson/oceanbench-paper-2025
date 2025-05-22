# oceanbench-paper-2025




**Create a new environemnt**

```bash
conda env create -f environment.yaml -n base
```


**Add to old environment**



```bash
conda env update -f environment.yaml -n base
```


---
## Run Scripts for Plots


#### **Forecast Maps**

```bash
bash scripts/plot_maps.sh
```

---
#### **PSD Plots** (Zonal Longitude + SpaceTime)


```bash
bash scripts/process_psd.sh --region=gulfstreamsubset
bash scripts/process_psd.sh --region=globe
```

**WARNING**: This script takes a very long time to run due to the PSD decomposition.