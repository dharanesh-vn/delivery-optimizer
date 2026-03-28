"""
Delivery Optimization System
==============================
Assigns deliveries from a CSV file to 3 agents while:
  - Respecting delivery priority (High > Medium > Low)
  - Balancing total distance per agent using a greedy LPT heuristic

Usage:
    python delivery_optimizer.py
    python delivery_optimizer.py --input my_data.csv --agents 3 --output plan.csv
"""

import heapq
import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_RANK = {"High": 1, "Medium": 2, "Low": 3}

VALID_PRIORITIES = set(PRIORITY_RANK.keys())

DEFAULT_INPUT  = "sample_deliveries.csv"
DEFAULT_OUTPUT = "delivery_plan.csv"
DEFAULT_AGENTS = 3


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_and_validate(filepath: str) -> pd.DataFrame:
    """
    Load the CSV and validate required columns and data types.
    Returns a clean DataFrame ready for processing.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Strip column name whitespace (common CSV issue)
    df.columns = df.columns.str.strip()

    required_columns = {"Location ID", "Distance from warehouse", "Delivery Priority"}
    missing = required_columns - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        sys.exit(1)

    # Drop rows with any null in required columns
    before = len(df)
    df = df.dropna(subset=list(required_columns))
    dropped = before - len(df)
    if dropped:
        print(f"[WARNING] Dropped {dropped} row(s) with missing values.")

    # Validate priority values
    df["Delivery Priority"] = df["Delivery Priority"].str.strip().str.capitalize()
    # Capitalize handles "high" -> "High", "HIGH" -> "High"
    df["Delivery Priority"] = df["Delivery Priority"].apply(
        lambda x: x[0].upper() + x[1:].lower() if isinstance(x, str) else x
    )
    invalid_priorities = ~df["Delivery Priority"].isin(VALID_PRIORITIES)
    if invalid_priorities.any():
        bad = df.loc[invalid_priorities, "Delivery Priority"].unique()
        print(f"[ERROR] Invalid priority values found: {list(bad)}")
        print(f"        Valid values are: {list(VALID_PRIORITIES)}")
        sys.exit(1)

    # Validate distance is numeric and non-negative
    df["Distance from warehouse"] = pd.to_numeric(
        df["Distance from warehouse"], errors="coerce"
    )
    non_numeric = df["Distance from warehouse"].isna()
    if non_numeric.any():
        print(f"[ERROR] Non-numeric distances found in {non_numeric.sum()} row(s).")
        sys.exit(1)

    negative_dist = df["Distance from warehouse"] < 0
    if negative_dist.any():
        print(f"[WARNING] {negative_dist.sum()} row(s) have negative distance — setting to 0.")
        df.loc[negative_dist, "Distance from warehouse"] = 0

    # Add numeric priority rank for sorting
    df["Priority Rank"] = df["Delivery Priority"].map(PRIORITY_RANK)

    print(f"[INFO] Loaded {len(df)} valid deliveries from '{filepath}'.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_for_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by:
      1. Priority rank ascending  (High=1 first)
      2. Distance descending within each priority group
         (LPT heuristic: assign larger distances first for better balance)
    """
    return df.sort_values(
        by=["Priority Rank", "Distance from warehouse"],
        ascending=[True, False]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assignment — Greedy LPT with min-heap
# ---------------------------------------------------------------------------

def assign_deliveries(df: pd.DataFrame, num_agents: int = DEFAULT_AGENTS) -> pd.DataFrame:
    """
    Greedy Longest Processing Time (LPT) assignment.

    Algorithm:
      - Maintain a min-heap of (total_distance, agent_id)
      - For each delivery (largest distance first within priority),
        pop the agent with the smallest current total distance,
        assign the delivery to that agent, push back with updated total.

    Time complexity: O(n log k)  where n = deliveries, k = agents
    Space complexity: O(n + k)
    """
    # Initialize heap: (total_distance, agent_id) for each agent
    heap = [(0.0, agent_id) for agent_id in range(1, num_agents + 1)]
    heapq.heapify(heap)

    assigned_agents = []

    for _, row in df.iterrows():
        # Pick the agent with least total distance
        total_dist, agent_id = heapq.heappop(heap)

        assigned_agents.append(f"Agent {agent_id}")

        # Update the agent's total distance and push back
        new_total = total_dist + row["Distance from warehouse"]
        heapq.heappush(heap, (new_total, agent_id))

    df = df.copy()
    df["Assigned Agent"] = assigned_agents
    return df


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def build_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-sort the assigned dataframe for a clean, readable output:
      Agent → Priority Rank → Distance ascending
    Returns only the columns needed for the output CSV.
    """
    output = df.sort_values(
        by=["Assigned Agent", "Priority Rank", "Distance from warehouse"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return output[[
        "Assigned Agent",
        "Location ID",
        "Delivery Priority",
        "Distance from warehouse",
    ]]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-agent summary: total distance, stop count.
    """
    summary = (
        df.groupby("Assigned Agent")
        .agg(
            Total_Distance=("Distance from warehouse", "sum"),
            Stops=("Location ID", "count"),
        )
        .reset_index()
        .rename(columns={"Assigned Agent": "Agent"})
        .sort_values("Agent")
    )
    return summary


def compute_imbalance(summary: pd.DataFrame) -> float:
    """
    Imbalance = (max_total - min_total) / max_total * 100
    Lower is better; 0% = perfectly balanced.
    """
    max_d = summary["Total_Distance"].max()
    min_d = summary["Total_Distance"].min()
    if max_d == 0:
        return 0.0
    return round((max_d - min_d) / max_d * 100, 2)


def print_report(output_df: pd.DataFrame, summary: pd.DataFrame, imbalance: float):
    """Print a formatted delivery plan and agent summary to the console."""
    sep = "=" * 62

    print(f"\n{sep}")
    print("  DELIVERY OPTIMIZATION REPORT")
    print(sep)

    for agent in summary["Agent"]:
        agent_df = output_df[output_df["Assigned Agent"] == agent]
        total = summary.loc[summary["Agent"] == agent, "Total_Distance"].values[0]
        stops = summary.loc[summary["Agent"] == agent, "Stops"].values[0]

        print(f"\n  {agent}  |  Stops: {stops}  |  Total Distance: {total:.1f} km")
        print("  " + "-" * 58)
        print(f"  {'Location ID':<14} {'Priority':<12} {'Distance (km)':>13}")
        print("  " + "-" * 58)
        for _, row in agent_df.iterrows():
            print(
                f"  {row['Location ID']:<14} "
                f"{row['Delivery Priority']:<12} "
                f"{row['Distance from warehouse']:>13.1f}"
            )

    print(f"\n{sep}")
    print("  AGENT SUMMARY")
    print(sep)
    print(f"  {'Agent':<12} {'Stops':>6} {'Total Distance (km)':>20}")
    print("  " + "-" * 40)
    for _, row in summary.iterrows():
        print(f"  {row['Agent']:<12} {int(row['Stops']):>6} {row['Total_Distance']:>20.1f}")

    max_d = summary["Total_Distance"].max()
    min_d = summary["Total_Distance"].min()
    avg_d = summary["Total_Distance"].mean()
    print("  " + "-" * 40)
    print(f"\n  Max distance : {max_d:.1f} km")
    print(f"  Min distance : {min_d:.1f} km")
    print(f"  Avg distance : {avg_d:.1f} km")
    print(f"  Imbalance    : {imbalance:.2f}%  (lower = better balance)")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Delivery Optimization — assigns deliveries to agents balancing total distance."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to input CSV file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Path to output CSV file (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=DEFAULT_AGENTS,
        help=f"Number of delivery agents (default: {DEFAULT_AGENTS})"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.agents < 1:
        print("[ERROR] Number of agents must be at least 1.")
        sys.exit(1)

    # 1. Load & validate
    df = load_and_validate(args.input)

    # 2. Sort for assignment (priority asc, distance desc within group)
    sorted_df = sort_for_assignment(df)

    # 3. Assign using greedy LPT min-heap
    assigned_df = assign_deliveries(sorted_df, num_agents=args.agents)

    # 4. Build clean output dataframe
    output_df = build_output_df(assigned_df)

    # 5. Compute summary & imbalance metric
    summary = compute_summary(output_df)
    imbalance = compute_imbalance(summary)

    # 6. Save output CSV
    output_df.to_csv(args.output, index=False)
    print(f"[INFO] Delivery plan saved to '{args.output}'.")

    # 7. Print console report
    print_report(output_df, summary, imbalance)


if __name__ == "__main__":
    main()
