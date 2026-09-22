# Beyond Travel Time: Multi-Objective Route Optimization using Symbolic Regression

**Azain Ayub - MSc Artificial Intelligence, Aston University**  
Supervised by (https://research.aston.ac.uk/en/persons/alina-patelli/)[Dr. Alina Patelli]

---

## Overview

This repository contains the three Jupyter notebooks that form the full implementation pipeline for the dissertation. The project learns interpretable symbolic-regression equations for traffic speed, NO2 and PM2.5 from real Chicago sensor data, uses them to score every road segment in the city's street network, and then returns a set of non-dominated route alternatives so a user can choose their own pollution–time trade-off.

All three notebooks are designed to run sequentially on **Google Colab**. Each exports the files the next one needs, so nothing is re-fetched or re-trained unnecessarily.

---

## Repository structure

```
dissertation_data.ipynb       Notebook 1 - data collection, spatial join, feature engineering
dissertation_sr.ipynb         Notebook 2 - symbolic regression model training and evaluation
dissertation_routing.ipynb    Notebook 3 - route scoring, Pareto search, comparative evaluation
README.md                     This file
```

---

## Notebooks

### `dissertation_data.ipynb` - Data Collection & Objective 1

Fetches one week of Chicago traffic and air-quality data, joins them spatially, builds the modelling features, and exports everything Notebook 2 needs.

**Data sources (both free, no API key required)**

| Dataset | Socrata ID | Coverage |
|---|---|---|
| Chicago Traffic Tracker (GPS speeds) | `4g9f-3jbs` | 1,039 arterial segments, 10-min intervals |
| Open Air Chicago (NO2 + PM2.5) | `xfya-dxtq` | 277 sensors, ~1.4 km grid |

**Analysis week:** 6–12 October 2025 (Monday-Sunday)

**What it does**
1. Fetches traffic data in two batches via the Socrata REST API.
2. Fetches Open Air Chicago hourly sensor readings.
3. Exploratory data analysis on each dataset separately.
4. Spatial join: matches each traffic segment to its nearest OAC sensor (within 2,000 m); gap-fills unmatched segments by sampling from the real OAC hourly distribution.
5. Feature engineering: temporal features (hour, day-of-week, cyclic encodings), lag features (speed and pollution one-to-three steps back).
6. Extends the Objective 1 coverage check to the full OSMnx road network.
7. Exports the modelling dataset and historical averages to `/content/`

**Files exported to `/content/`**

| File | Used by |
|---|---|
| `chicago_modelling_dataset.csv` | Notebook 2 |
| `chicago_meta.json` | Notebook 2 |
| `chicago_hist_averages.csv` | Notebooks 2 and 3 |
| `chicago_oac_coverage_map.html` | (standalone map) |

**Objective 1 result:** 100% coverage of 1,039 traffic segments (98.6% from real sensors, 1.4% synthetic gap-fill)

---

### `dissertation_sr.ipynb` - Symbolic Regression Models & Objective 2

Trains three symbolic-regression models (one per target: speed, NO2, PM2.5) using DEAP's genetic programming with NSGA-II multi-objective selection, then evaluates them against Linear Regression and Random Forest baselines.

**Reproducibility**

Everything is seeded from a single constant:

```python
MASTER_SEED = 42
```

Library versions are pinned in the install cell (`deap==1.4.4`, `scikit-learn==1.8.0`) so the same expressions are recovered on re-run.

**Per-model structural priors**

| Model | Operators | Max depth | Rationale |
|---|---|---|---|
| Speed | add, sub, mul, neg, sqrt, sin, cos | 6 | Smooth multiplicative relationship with recent speed; no division to avoid instability |
| NO2 | All (including div, log) | 8 | Non-linear emission curve expected; full operator set needed |
| PM2.5 | add, sub, mul, neg, sqrt, sin, cos | 5 | Most persistence-dominated target; conservative prior to avoid overfitting noise |

**Search parameters:** population 300, generations 50, 3 multi-start restarts

**Results**

| Model | Expression | Nodes | R2 | Status |
|---|---|---|---|---|
| Speed | `√(spd_lag1 x spd_lag2)` | 4 | 0.533 | Below 0.70 target |
| NO2 | `no2_lag1 + sin(sin(-sin(dow_cos/is_rush))) × (−hour_cos)` | 12 | 0.911 | Met |
| PM2.5 | `pm25_lag1 x 0.97` | 3 | 0.879 | Met |

**Files exported to `/content/`**

| File | Used by |
|---|---|
| `sr_models.pkl` | Notebook 3 |
| `sr_model_spd.pkl`, `sr_model_no2.pkl`, `sr_model_pm25.pkl` | Notebook 3 (individual) |

---

### `dissertation_routing.ipynb` - Route Scoring & Objectives 3 & 4

Loads the trained SR models and the Chicago road network, scores every edge at each departure time using the SR expressions, and runs a weighted-sum NSGA-II search to find non-dominated route alternatives. Evaluates across 288 scenarios (32 journeys x 9 departure times).

**Road network:** Chicago drive graph via OSMnx - 52,273 nodes, 141,466 edges

**Experimental design**

- 32 origin-destination pairs spanning central, residential and airport journeys
- 9 departure times covering AM rush, PM rush, midday, evening and night slots
- Each scenario scored against a pure shortest-time Dijkstra baseline

**Objective 3 result:** 196 / 288 scenarios (68%) returned >= 3 distinct non-dominated routes (mean 4.0 per scenario)

**Objective 4 results**

| Metric | Mean pollution reduction | Mean time increase | Scenarios meeting target |
|---|---|---|---|
| Absolute (unconstrained) | 23.2% | 27.9% | 49 / 288 (17%) |
| Practical (<= 20% time budget) | 18.3% | 7.7% | 133 / 288 (46%) |

**Files exported to `/content/`**

| File | Description |
|---|---|
| `pareto_grid.png` | Pareto fronts for all 32 journeys |
| `o4_aggregate.png` | O4 scatter across all 288 scenarios |
| `multi_journey_map.html` | Interactive Folium map of all routes |

---

## How to run

### Prerequisites

- A Google account (to use Colab) or a local Python 3.10+ environment
- No API keys required - both data sources are fully open

### Step-by-step

1. Open `dissertation_data.ipynb` in Google Colab.
2. Run all cells top to bottom - takes roughly 10-15 minutes (mostly API fetch time).
3. Confirm `/content/chicago_modelling_dataset.csv` exists before moving on.
4. Open `dissertation_sr.ipynb` in a **new Colab tab** (so `/content/` is shared).
5. Run all cells — training takes 30–60 minutes depending on Colab resources.
6. Confirm `/content/sr_models.pkl` exists before moving on.
7. Open `dissertation_routing.ipynb` in the **same Colab session** or upload the pkl/csv files manually.
8. Run all cells - edge scoring takes roughly 4 minutes per departure time (9 x 4 min roughly 36 min), then route search

> **Important:** Steps 4-6 must run in the same Colab session as steps 1-3, or you must manually upload `chicago_modelling_dataset.csv`, `chicago_meta.json` and `chicago_hist_averages.csv` to `/content/` before running Notebook 2. Colab's `/content/` directory resets when a session ends.

### Running locally instead of Colab

Replace any `/content/` paths with your local working directory, then install dependencies:

```bash
pip install osmnx geopandas folium scikit-learn seaborn deap==1.4.4 tqdm pymoo
```

---

## Dependencies

| Library | Version | Used in |
|---|---|---|
| `deap` | 1.4.4 (pinned) | Notebooks 2, 3 |
| `scikit-learn` | 1.8.0 (pinned) | Notebooks 1, 2, 3 |
| `osmnx` | latest | Notebooks 1, 3 |
| `geopandas` | latest | Notebook 1 |
| `folium` | latest | Notebooks 1, 3 |
| `seaborn` | latest | Notebooks 1, 2 |
| `tqdm` | latest | Notebooks 2, 3 |
| `pymoo` | latest | Notebook 3 |
| `networkx` | (via osmnx) | Notebook 3 |

DEAP and scikit-learn are pinned because a version change can produce different GP expressions even with an identical random seed.

---

## Key design decisions

**Why three separate notebooks rather than one script?**  
Each notebook exports its outputs to disk, so any stage can be re-run in isolation without repeating the stages before it. This made debugging tractable - a bug in the SR training (for example, the input-feature mismatch found during Sprint 2) could be fixed and re-run without re-fetching 665,000 rows of traffic data.

**Why DEAP rather than PySR or gplearn?**  
DEAP gives full control over which mathematical operators each model is allowed to use, which is what makes the per-model structural priors possible. PySR and gplearn do not expose this level of per-primitive control.

**Why the `chicago_hist_averages.csv` file?**  
SR models are trained on lag features (last observed speed, NO2 and PM2.5). At routing time there is no live reading, so each edge's lag features are substituted with the historical mean for that segment, hour and day-of-week. This file is pre-computed in Notebook 1 and reused in Notebooks 2 and 3.

**Why seed DEAP's random state from `MASTER_SEED`?**  
GP is non-deterministic. Fixing the seed means that re-running the notebook on the same data produces the same expressions. If the seed is changed or a different library version is installed, different (but comparably accurate) expressions may result - this is documented in the dissertation.

---

## Citation

> Ayub, A. (2026) *Beyond Travel Time: Multi-Objective Route Optimization using Symbolic Regression*. MSc dissertation, Aston University.

---

## Licence

Code released for academic reproducibility. Data remains subject to the City of Chicago Open Data Licence (<https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html>).