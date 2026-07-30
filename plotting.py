"""
Pole Time Prediction Visualization

Creates visual comparisons of Formula 1 pole lap times using 2025
historical results, 2026 model predictions, and available 2026
qualifying results. Includes a summary table, trend comparison plot,
and feature importance visualization to interpret the machine learning model.

Author: Jazib Ahmed
"""

import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from fastf1.plotting import get_team_color
import fastf1.plotting as f1plot

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60

    return f"{minutes}:{seconds:06.3f}" if minutes > 0 else f"{seconds:.3f}"

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#2b2b2b'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['savefig.facecolor'] = '#2b2b2b'


poletimes = pd.read_csv("datasets/future_predictions.csv")
poleset = pd.read_csv("datasets/pole_mergeset.csv")

poleset2025 = poleset[poleset["Year"] == 2025]
poleset2025 = poleset2025[poleset2025["Race"].isin(poletimes["Race"])]

poleset2026 = pd.DataFrame({
    'Race': ['Belgian Grand Prix', 'Hungarian Grand Prix'],
    'PoleTime': [104.361, 77.207]
})


# ==========================
# FIGURE 1: TABLE
# ==========================

fig1, ax1 = plt.subplots(figsize=(12, 8))

ax1.axis("off")

table_data = []

for race in poletimes['Race']:

    pole_2025 = poleset2025.loc[
        poleset2025["Race"] == race, "PoleTime"
    ]

    actual_2026 = poleset2026.loc[
        poleset2026["Race"] == race, "PoleTime"
    ]

    predicted_2026 = poletimes.loc[
        poletimes["Race"] == race, "PredictedPoleTime"
    ].iloc[0]

    table_data.append([
        race,
        format_time(pole_2025.iloc[0]) if len(pole_2025) else "N/A",
        format_time(predicted_2026),
        format_time(actual_2026.iloc[0]) if len(actual_2026) else "TBD"
    ])


table = ax1.table(
    cellText=table_data,
    colLabels=[
        "Race",
        "2025 Pole Time",
        "2026 Predicted Time",
        "2026 Actual Time"
    ],
    loc="center",
    cellLoc="center",
    colLoc="center"
)

# Larger font for 12x8 figure
table.auto_set_font_size(False)
table.set_fontsize(16)

# Increase row height
table.scale(1, 2.5)

# Adjust column widths
for (row, col), cell in table.get_celld().items():
    if col == 0:
        cell.set_width(0.35)  # Race
    else:
        cell.set_width(0.22)  # Time columns

    cell.set_facecolor("#2b2b2b")
    cell.set_edgecolor("#555555")
    cell.get_text().set_color("white")

    if row == 0:
        cell.set_facecolor("#1a1a1a")
        cell.get_text().set_weight("bold")
        cell.get_text().set_fontsize(14)

for (row, col), cell in table.get_celld().items():

    cell.set_edgecolor("#555555")
    cell.get_text().set_color("white")

    # Header row
    if row == 0:
        header_colors = [
            "#333333",  # Race
            "#1f3325",  # 2025 historical
            "#3b1f1f",  # Prediction
            "#1f293b"   # Actual
        ]

        cell.set_facecolor(header_colors[col])
        cell.get_text().set_weight("bold")
        cell.get_text().set_fontsize(14)

    else:
        cell.set_facecolor("#2b2b2b")

        # Highlight missing/future values
        text = cell.get_text().get_text()

        if text == "N/A":
            cell.set_facecolor("#3a3a3a")
            cell.get_text().set_color("#bbbbbb")

        elif text == "TBD":
            cell.set_facecolor("#3a3020")
            cell.get_text().set_color("#ffd27f")


ax1.set_title(
    "Formula 1 Pole Lap Time ML Predictions",
    fontsize=22,
    color="white",
    pad=30
)

fig1.text(
    0.98,
    0.02,
    "* All 2026 predictions are in dry conditions",
    ha="right",
    fontsize=10,
    color="white"
)


plt.tight_layout()

fig1.savefig(
    "images/pole_prediction_table.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# ==========================
# FIGURE 2: GRAPH
# ==========================

fig2, ax2 = plt.subplots(figsize=(12, 8))

ax2.plot(
    poletimes['Race'],
    poletimes['PredictedPoleTime'],
    marker='s',
    color='red',
    linewidth=2,
    linestyle='--',
    alpha=0.8,
    label='2026 Prediction'
)

ax2.plot(
    poleset2025['Race'],
    poleset2025['PoleTime'],
    marker='^',
    color='green',
    linewidth=2,
    alpha=0.8,
    label='2025 Qualifying'
)

ax2.scatter(
    poleset2026['Race'],
    poleset2026['PoleTime'],
    marker='o',
    color='blue',
    edgecolor='white',
    alpha=1,
    s=80,
    label='2026 Qualifying'
)


ax2.set_title(
    "Pole Laptime ML Predictor (2025 vs 2026 vs Predicted)",
    fontsize=22,
    color="white"
)

ax2.set_ylabel(
    "Pole Time (s)",
    fontsize=16,
    color="white"
)

race_labels = [race.replace("Grand Prix", "GP") for race in poletimes["Race"]]
ax2.set_xticks(poletimes['Race'])
ax2.set_xticklabels(
    race_labels,
    rotation=45,
    fontsize=15,
    ha='right',
    color="white"
)

ax2.tick_params(axis='y', labelsize=15, colors='white')
ax2.tick_params(colors="white")

ax2.grid(True, alpha=0.2, color="gray")

ax2.legend(
    loc="upper right",
    facecolor="#1a1a1a",
    edgecolor="#444444",
    labelcolor="white",
    framealpha = 0.2,
    fontsize=12.5
)

plt.tight_layout()

fig2.savefig(
    "images/pole_prediction_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================
# FIGURE 3: FEATURE IMPORTANCE
# ==========================

importance = pd.read_csv("datasets/feature_importance.csv")

# Sort and select top features
importance = importance.sort_values(
    by="Importance",
    ascending=True
)

top_features = importance.tail(15)


fig3, ax3 = plt.subplots(figsize=(12, 8))

ax3.barh(
    top_features["Feature"],
    top_features["Importance"],
    alpha=0.85
)

ax3.set_title(
    "Pole Prediction Feature Importance",
    fontsize=20,
    color="white"
)

ax3.set_xlabel(
    "Importance (%)",
    fontsize=15,
    color="white"
)

ax3.tick_params(axis='y', labelsize=13, colors='white')
ax3.tick_params(axis='x', labelsize=15, colors='white')

ax3.grid(
    axis="x",
    alpha=0.2,
    color="gray"
)

# Add importance values beside bars
for i, value in enumerate(top_features["Importance"]):
    ax3.text(
        value*1.002,
        i,
        f"{value:.2f}",
        va="center",
        color="white",
        fontsize=13
    )

max_importance = top_features["Importance"].max()
ax3.set_xlim(0, max_importance *1.15)

plt.tight_layout()

fig3.savefig(
    "images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()