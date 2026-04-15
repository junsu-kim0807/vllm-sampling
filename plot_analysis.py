# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# %%
# ============================================================
# User config
# ============================================================
BATCH_PT_PATH = Path("./results/spec_decode/batch_request_characteristics.pt")
PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METHOD = "speculative"
ROUND_LIMIT = 20
CONF_BINS = np.linspace(0.0, 1.0, 11)

# If None, use all datasets available for METHOD
DATASET_ORDER = None

# %%
# ============================================================
# Style
# ============================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rc("font", size=12)

sns.set_palette("tab10")
sns.set_style("whitegrid")
palette = sns.color_palette("tab10")

BLUE = palette[0]
ORANGE = palette[1]
GREEN = palette[2]
RED = palette[3]

# %%
# ============================================================
# Helpers
# ============================================================
def load_batch_artifact(pt_path: Path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    records = obj["records"]
    df = pd.DataFrame(records)
    return obj, df


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def filter_method_df(df, method=METHOD):
    return df[df["method"] == method].copy()


def get_dataset_order(df, preferred=None):
    available = sorted(df["dataset"].dropna().unique().tolist())
    if preferred is None:
        return available
    return [x for x in preferred if x in available]


def get_single_record(df, dataset, batch_size, num_spec_tokens=None):
    sub = df[
        (df["dataset"] == dataset) &
        (df["batch_size"] == batch_size)
    ].copy()

    if num_spec_tokens is not None:
        sub = sub[sub["num_speculative_tokens"] == num_spec_tokens].copy()

    if len(sub) == 0:
        return None

    sub = sub.sort_values(["num_speculative_tokens", "selected_worker_id", "selected_worker_file"])
    return sub.iloc[0].to_dict()


def choose_representative_request_trace(record, round_limit=ROUND_LIMIT):
    rows = read_jsonl(record["selected_worker_file"])
    req_to_rows = {}

    for row in rows:
        rid = row["req_id"]
        req_to_rows.setdefault(rid, []).append(row)

    best_req_id = None
    best_trace = None

    for rid, trace_rows in sorted(req_to_rows.items()):
        trace_rows = sorted(trace_rows, key=lambda x: int(x["step_id"]))
        if len(trace_rows) >= round_limit:
            best_req_id = rid
            best_trace = trace_rows[:round_limit]
            break

    if best_trace is None:
        ranked = sorted(
            req_to_rows.items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )
        best_req_id, trace_rows = ranked[0]
        best_trace = sorted(trace_rows, key=lambda x: int(x["step_id"]))[:round_limit]

    trace_df = pd.DataFrame(best_trace).copy()
    trace_df["round_index"] = np.arange(len(trace_df))
    trace_df["acceptance_length"] = trace_df["num_accepted_tokens"].astype(int)
    trace_df["is_misspec"] = trace_df["acceptance_length"] == 0

    return best_req_id, trace_df


def histogram_to_probability(hist_dict):
    if hist_dict is None:
        return pd.DataFrame(columns=["acceptance_length", "probability", "count"])
    items = sorted((int(k), int(v)) for k, v in hist_dict.items())
    total = sum(v for _, v in items)
    if total == 0:
        return pd.DataFrame(columns=["acceptance_length", "probability", "count"])
    return pd.DataFrame({
        "acceptance_length": [k for k, _ in items],
        "count": [v for _, v in items],
        "probability": [v / total for _, v in items],
    })


def extract_match_probabilities(record):
    topk = record["topk_match_probability_under_misspeculation"]
    return pd.DataFrame({
        "topk": ["top_2", "top_3", "top_4", "top_5"],
        "probability": [
            topk.get("top_2", {}).get("probability", None),
            topk.get("top_3", {}).get("probability", None),
            topk.get("top_4", {}).get("probability", None),
            topk.get("top_5", {}).get("probability", None),
        ],
    })


def pearson_corr_safe(x, y):
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_confidence_mismatch_df(record):
    rows = read_jsonl(record["selected_worker_file"])
    out = []

    for row in rows:
        confs = row.get("first_draft_topk_confidences")
        if not isinstance(confs, list) or len(confs) == 0:
            continue

        first_conf = confs[0]
        if first_conf is None:
            continue

        acceptance_length = int(row.get("num_accepted_tokens", 0))
        out.append({
            "confidence": float(first_conf),
            "mismatch": 1 if acceptance_length == 0 else 0,
            "acceptance_length": acceptance_length,
            "step_id": int(row.get("step_id", -1)),
            "req_id": row.get("req_id"),
        })

    return pd.DataFrame(out)


def build_binned_mismatch_curve(df_conf, bins=CONF_BINS):
    if len(df_conf) == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "mismatch_probability", "count"])

    df_conf = df_conf.copy()
    df_conf["bin"] = pd.cut(df_conf["confidence"], bins=bins, include_lowest=True, right=True)

    grouped = (
        df_conf.groupby("bin", observed=False)
        .agg(
            mismatch_probability=("mismatch", "mean"),
            count=("mismatch", "size"),
        )
        .reset_index()
    )

    grouped["bin_left"] = grouped["bin"].apply(lambda x: float(x.left))
    grouped["bin_right"] = grouped["bin"].apply(lambda x: float(x.right))
    grouped["bin_center"] = (grouped["bin_left"] + grouped["bin_right"]) / 2.0
    return grouped


# %%
# ============================================================
# Load
# ============================================================
artifact, batch_df = load_batch_artifact(BATCH_PT_PATH)
batch_df = filter_method_df(batch_df, METHOD)
datasets = get_dataset_order(batch_df, preferred=DATASET_ORDER)

print("artifact type:", artifact["type"])
print("available methods:", artifact["stats"]["methods"])
print("available datasets for method:", datasets)
print("available batch sizes:", sorted(batch_df["batch_size"].dropna().unique().tolist()))
print("available num_speculative_tokens:", sorted(batch_df["num_speculative_tokens"].dropna().unique().tolist()))

# %%
# ============================================================
# Figure 1
# batch_size = 1
# One representative request trace per dataset up to 20 rounds
# ============================================================
records_bs1 = {}
for dataset in datasets:
    rec = get_single_record(batch_df, dataset=dataset, batch_size=1)
    if rec is not None:
        records_bs1[dataset] = rec

ncols = max(1, len(records_bs1))
fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5), squeeze=False)
axes = axes[0]

for ax, dataset in zip(axes, records_bs1.keys()):
    req_id, trace_df = choose_representative_request_trace(records_bs1[dataset], round_limit=ROUND_LIMIT)

    ax.plot(
        trace_df["round_index"],
        trace_df["acceptance_length"],
        marker="o",
        linewidth=1.6,
        markersize=5,
        color=BLUE,
        label=dataset,
    )

    miss = trace_df[trace_df["is_misspec"]]
    if len(miss) > 0:
        ax.scatter(
            miss["round_index"],
            miss["acceptance_length"],
            color="red",
            s=45,
            zorder=5,
        )

    max_accept = int(trace_df["acceptance_length"].max()) if len(trace_df) else 1
    ax.set_title(dataset)
    ax.set_xlabel("Verification Round")
    ax.set_ylabel("Acceptance Length")
    ax.set_xticks(np.arange(0, ROUND_LIMIT + 1, 5))
    ax.set_ylim(0, max(5, max_accept) + 0.2)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=10, frameon=True, fancybox=True)

plt.tight_layout()
fig.savefig(PLOTS_DIR / f"batchchar_fig1_acceptance_trace_{METHOD}.pdf")
fig.savefig(PLOTS_DIR / f"batchchar_fig1_acceptance_trace_{METHOD}.png", dpi=600)
plt.show()

# %%
# ============================================================
# Figure 2
# batch_size = 1
# misspeculation probability vs num_speculative_tokens
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

for dataset in datasets:
    sub = (
        batch_df[
            (batch_df["dataset"] == dataset) &
            (batch_df["batch_size"] == 1)
        ]
        .sort_values("num_speculative_tokens")
    )
    if len(sub) == 0:
        continue

    ax.plot(
        sub["num_speculative_tokens"],
        sub["misspeculation_probability"],
        marker="o",
        linewidth=1.6,
        markersize=5,
        label=dataset,
    )

ax.set_xlabel("Number of Draft Tokens")
ax.set_ylabel("Probability of Misspeculation")
ax.grid(True, axis="both", linestyle=":", linewidth=0.8)
ax.legend(loc="upper left", ncol=max(1, min(4, len(datasets))), frameon=True, fancybox=True, fontsize=10)

plt.tight_layout()
fig.savefig(PLOTS_DIR / f"batchchar_fig2_misspec_vs_k_{METHOD}.pdf")
fig.savefig(PLOTS_DIR / f"batchchar_fig2_misspec_vs_k_{METHOD}.png", dpi=600)
plt.show()

# %%
# ============================================================
# Figure 3
# batch_size = 256
# acceptance length distribution per dataset
# ============================================================
records_bs256 = {}
for dataset in datasets:
    rec = get_single_record(batch_df, dataset=dataset, batch_size=256)
    if rec is not None:
        records_bs256[dataset] = rec

ncols = max(1, len(records_bs256))
fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 2.6), squeeze=False)
axes = axes[0]

for ax, dataset in zip(axes, records_bs256.keys()):
    hist_df = histogram_to_probability(records_bs256[dataset]["acceptance_length_histogram_global"])
    ax.plot(
        hist_df["acceptance_length"],
        hist_df["probability"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        color=BLUE,
    )
    ax.set_title(dataset)
    ax.set_xlabel("Acceptance Length")
    ax.set_ylabel("Probability of Request\nbeing in a batch")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.8)

plt.tight_layout()
fig.savefig(PLOTS_DIR / f"batchchar_fig3_acceptance_dist_b256_{METHOD}.pdf")
fig.savefig(PLOTS_DIR / f"batchchar_fig3_acceptance_dist_b256_{METHOD}.png", dpi=600)
plt.show()

# %%
# ============================================================
# Figure 4
# batch_size = 1
# target match probability under misspeculation for top-2,3,4,5
# ============================================================
match_rows = []
for dataset in datasets:
    rec = get_single_record(batch_df, dataset=dataset, batch_size=1)
    if rec is None:
        continue
    sub = extract_match_probabilities(rec)
    sub["dataset"] = dataset
    match_rows.append(sub)

match_df = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(datasets))
bar_width = 0.18

topk_order = ["top_2", "top_3", "top_4", "top_5"]
offsets = {
    "top_2": -1.5 * bar_width,
    "top_3": -0.5 * bar_width,
    "top_4": 0.5 * bar_width,
    "top_5": 1.5 * bar_width,
}
color_map = {
    "top_2": BLUE,
    "top_3": ORANGE,
    "top_4": GREEN,
    "top_5": RED,
}
label_map = {
    "top_2": "Top-2",
    "top_3": "Top-3",
    "top_4": "Top-4",
    "top_5": "Top-5",
}

for tk in topk_order:
    ys = []
    for dataset in datasets:
        val = match_df[
            (match_df["dataset"] == dataset) &
            (match_df["topk"] == tk)
        ]["probability"]
        ys.append(val.iloc[0] if len(val) > 0 else np.nan)

    ax.bar(
        x + offsets[tk],
        ys,
        width=bar_width,
        color=color_map[tk],
        edgecolor="black",
        linewidth=0.8,
        label=label_map[tk],
    )

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Target Match Probability")
ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
ax.grid(False, axis="x")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=True, fancybox=True)

plt.tight_layout()
fig.savefig(PLOTS_DIR / f"batchchar_fig4_target_match_{METHOD}.pdf")
fig.savefig(PLOTS_DIR / f"batchchar_fig4_target_match_{METHOD}.png", dpi=600)
plt.show()

# %%
# ============================================================
# Figure 5
# batch_size = 1
# mismatch probability vs first draft token confidence
# ============================================================
corr_rows = []
curve_dict = {}

for dataset in datasets:
    rec = get_single_record(batch_df, dataset=dataset, batch_size=1)
    if rec is None:
        continue

    conf_df = build_confidence_mismatch_df(rec)
    curve_df = build_binned_mismatch_curve(conf_df, bins=CONF_BINS)
    corr = pearson_corr_safe(conf_df["confidence"], conf_df["mismatch"]) if len(conf_df) else np.nan

    curve_dict[dataset] = curve_df
    corr_rows.append({
        "dataset": dataset,
        "num_points": len(conf_df),
        "pearson_corr": corr,
    })

corr_df = pd.DataFrame(corr_rows)
print("Confidence vs mismatch correlation")
print(corr_df)

ncols = max(1, len(curve_dict))
fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.2), squeeze=False)
axes = axes[0]

for ax, dataset in zip(axes, curve_dict.keys()):
    curve_df = curve_dict[dataset]
    row = corr_df[corr_df["dataset"] == dataset].iloc[0]
    corr = row["pearson_corr"]

    ax.plot(
        curve_df["bin_center"],
        curve_df["mismatch_probability"],
        marker="o",
        linewidth=1.5,
        markersize=4,
        color=BLUE,
    )
    ax.set_title(f"{dataset}\nPearson r = {corr:.3f}" if pd.notna(corr) else f"{dataset}\nPearson r = N/A")
    ax.set_xlabel("First Draft Token Confidence")
    ax.set_ylabel("Mismatch Probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.8)

plt.tight_layout()
fig.savefig(PLOTS_DIR / f"batchchar_fig5_confidence_correlation_{METHOD}.pdf")
fig.savefig(PLOTS_DIR / f"batchchar_fig5_confidence_correlation_{METHOD}.png", dpi=600)
plt.show()
