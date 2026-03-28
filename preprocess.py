import pandas as pd

df = pd.read_csv("C:/Users/dhara/Downloads/delivery_optimizer/Food_Delivery_Times.csv")

# keep only what we need
result = df[["Order_ID", "Distance_km", "Traffic_Level"]].copy()

# rename to match optimizer's expected format
result.columns = ["Location ID", "Distance from warehouse", "Delivery Priority"]

# save
result.to_csv("deliveries.csv", index=False)