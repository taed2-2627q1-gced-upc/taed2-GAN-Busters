from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cargar datos
csv_path = Path("tables/class_proportions.csv")
df = pd.read_csv(csv_path)

# Filtrar por el check de subclases
subclass_df = df[df["check"] == "subclass_by_split"].copy()
subclass_df["percentage"] = subclass_df["proportion"] * 100
subclass_df["split"] = pd.Categorical(
    subclass_df["split"], categories=["train", "val", "test"], ordered=True
)

# 2. Generar gráfico
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 4.5))

sns.barplot(
    data=subclass_df,
    x="group",
    y="percentage",
    hue="split",
    palette=["#3470a3", "#f28e2b", "#59a14f"],
)

plt.axhline(
    10.0,
    color="gray",
    linestyle="--",
    linewidth=1,
    label="Theoretical uniform distribution (10%)",
)
plt.title(
    "Distribution of Subclasses across Partitions (CIFAKE)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Subclass", fontsize=10)
plt.ylabel("Proportion (%)", fontsize=10)
plt.ylim(8, 12)
plt.xticks(rotation=25)
plt.legend(title="Split", frameon=True)
plt.tight_layout()

# Guardar figura
out_fig = Path("figures/subclass_distribution_splits.png")
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300)
plt.close()