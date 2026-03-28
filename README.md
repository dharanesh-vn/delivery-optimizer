# Delivery Optimization System

A python based logistics tool that reads delivery data, sorts them by priority and assigns them to 3 agents while keeping the total distance as equal as possible between agents.

---

## What This Project Does

The idea is simple — we have a list of delivery orders, each with a distance and a priority level. The goal is to assign these deliveries to 3 agents in a way that no single agent ends up with way more distance than the others. Priority is always respected meaning High priority orders always go first before Medium or Low ones.

---

## Dataset

The dataset used here is taken from Kaggle — its a food delivery dataset that contains real order information.

**Source:** Food Delivery Times dataset from Kaggle
**Link:** https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction

The original dataset has these columns:
- Order_ID — unique id for each order
- Distance_km — how far the delivery is in kilometers
- Weather — weather condition at the time (Clear, Rainy, Foggy etc)
- Traffic_Level — traffic during delivery, either Low, Medium or High
- Time_of_Day — Morning, Afternoon, Evening or Night

Out of these we only use three columns for our task — Order_ID, Distance_km and Traffic_Level. The reason we use Traffic_Level as priority is because orders in high traffic areas take longer to reach even if the distance is similar, so they need to be dispatched earlier. This makes High traffic = High priority which is a logical and practical decision.

Total records in dataset: 1000 orders
After removing rows with missing values: 970 valid deliveries

---

## Files in This Project

**Food_Delivery_Times.csv**
This is the original dataset downloaded from Kaggle. We haven't modified this file at all.

**preprocess.py**
This script reads the original dataset and picks only the 3 columns we need. It renames them to match what our optimizer expects and saves the result as deliveries.csv. Its a short script but an important step in the pipeline.

**deliveries.csv**
This is the cleaned and renamed version of the kaggle data. It has exactly 3 columns — Location ID, Distance from warehouse and Delivery Priority. This file is the actual input to our optimizer.

**delivery_optimizer.py**
This is the main file. It reads deliveries.csv, sorts the orders by priority first and then assigns them to 3 agents using a greedy algorithm. The algorithm always gives the next delivery to whichever agent currently has the least total distance. This keeps the distribution balanced.

**delivery_plan.csv**
This is the output file. It shows which agent is assigned to which delivery, the priority and the distance. You can open this in excel to see the full plan clearly.

---

## How to Run

First make sure you have python installed and then install the only dependency:

```
pip install pandas
```

Then run in this order:

```
python preprocess.py
python delivery_optimizer.py --input deliveries.csv --output delivery_plan.csv
```

Thats it. The delivery_plan.csv will be generated with the full assignment.

You can also change the number of agents if needed:

```
python delivery_optimizer.py --input deliveries.csv --agents 4
```

---

## How the Algorithm Works

After loading the data we sort deliveries by priority — High comes first, then Medium, then Low. Within each priority group we sort by distance in descending order (largest first). This is based on a well known scheduling concept called LPT (Longest Processing Time) where assigning bigger tasks first leads to better overall balance.

For assigning deliveries we use a min-heap. We keep track of each agents total distance and always assign the next delivery to the agent with the lowest current total. This runs in O(n log k) time where n is number of deliveries and k is number of agents.

---

## Results

After running on the real Kaggle dataset:

| Agent | Stops | Total Distance |
|---|---|---|
| Agent 1 | 323 | 3244.8 km |
| Agent 2 | 323 | 3245.3 km |
| Agent 3 | 324 | 3245.3 km |

Imbalance came out to just **0.02%** which shows the algorithm is doing a very good job at balancing the load across all three agents.

---

## Assumptions

- Traffic_Level from the dataset is used as Delivery Priority since high traffic directly affects delivery time
- Distance is treated as the main metric for balancing, not number of stops
- Missing rows in the dataset are dropped with a warning before processing
