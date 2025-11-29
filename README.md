# Retail Sales Forecasting & Inventory Optimization System

This project is an **industry-oriented Data Science project** that simulates how
retailers forecast demand and decide how much inventory to order.

It predicts **daily store–item level sales** and converts those forecasts into:
- Safety Stock  
- Reorder Point  
- Suggested Order Quantity (using EOQ logic)

---

## 🔍 Problem Statement

Retailers lose money in two opposite ways:

- **Stockouts** → product not available when customer wants it → lost sales  
- **Overstock** → too much inventory sitting in warehouse → blocked capital & high holding cost  

The goal of this project is to:
1. Forecast future sales at **store × item × date** level.  
2. Use those forecasts to **derive inventory recommendations** that balance service level and cost.

---

## 🧠 Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn (RandomForestRegressor)
- matplotlib
- scipy
- joblib

---

## 📁 Project Structure

```bash
.
├─ data/
│  └─ retail_timeseries.csv
├─ outputs/
│  ├─ model/
│  │  └─ retail_forecast_model.pkl
│  ├─ figures/
│  │  └─ sample_actual_vs_pred.png
│  └─ logs/
│     └─ run_log.txt
├─ src/
│  └─ train_forecast_inventory.py
├─ .github/
│  └─ workflows/
│     └─ ci-basic.yml
├─ README.md
├─ requirements.txt
├─ LICENSE
└─ .venv/ (local env, not pushed to GitHub)
