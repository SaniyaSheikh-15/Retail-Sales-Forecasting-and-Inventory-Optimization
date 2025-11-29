## 🛒 Retail Sales Forecasting & Inventory Optimization System

A Data Science project simulating real-world Demand Forecasting + Inventory Replenishment in retail & D2C.
Forecasts item-level sales and turns those predictions into optimal order quantities using Safety Stock, Reorder Point, EOQ, and lead-time demand.

This is the same pipeline retailers like Reliance Retail, BigBasket, Flipkart, Amazon use to reduce stockouts & avoid overstock build-up.

## 📌 Objective

| Function                        | Output                              |
| ------------------------------- | ----------------------------------- |
| Forecasts store/SKU-level sales | Daily / weekly predictions          |
| Models forecast uncertainty     | Standard deviation of residuals     |
| Computes safety stock           | Based on service level targets      |
| Calculates reorder points       | To avoid stock-out during lead time |
| Suggests EOQ replenishment      | Cost-optimized purchase quantity    |
This delivers both demand planning + inventory decision automation — a complete DS + Ops workflow.

## 🌍 Industry Relevance

Retailers lose revenue to stock-outs & working capital due to overstock.
Demand forecasting + inventory science solves this.

This project models how enterprise supply chain teams operate:
data → forecasting → uncertainty modelling → inventory policy → dashboard/UI

## Used for:
Replenishment automation
Fill-rate improvement
Working-capital efficiency
Multi-SKU stocking strategy
D2C / FMCG / Grocery retail

## ⚙️ Tech Stack

| Component           | Tools                             |
| ------------------- | --------------------------------- |
| Data                | pandas, numpy                     |
| Forecasting Model   | RandomForestRegressor             |
| Feature Engineering | Rolling stats, lags, seasonality  |
| Inventory Science   | Safety Stock, ROP, EOQ            |
| UI / Deployment     | Streamlit                         |
| Mlops-ready         | Model save/load (joblib), logging |


## 🚀 Run the Project Locally
1️⃣ Create & activate environment
python -m venv .venv
.venv\Scripts\activate   # Windows

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train ML model + generate dataset
python src/train_forecast_inventory.py

Output will include:

✔ Dataset created → data/retail_timeseries.csv
✔ Trained model → outputs/model/*.pkl
✔ Visualization → outputs/figures/sample_actual_vs_pred.png
✔ Inventory recommendation in terminal

4️⃣ Launch the UI
streamlit run app_streamlit.py

Opens dashboard at:
http://localhost:8501


## ⭐ Contributions Welcome

Fork → Add new models → Open PR.
Ideas like reinforcement-learning reorder strategies or Bayesian forecasting are highly appreciated.
