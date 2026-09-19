import pandas as pd

def daily_counts(df, date_col="received_at", group_col="subcategory"):
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col])
    out = (
        temp.groupby([pd.Grouper(key=date_col, freq="D"), group_col])
        .size()
        .reset_index(name="count")
    )
    return out

def detect_surge(df, group_col="subcategory", recent_days=7, baseline_days=21):
    temp = df.copy()
    temp["received_at"] = pd.to_datetime(temp["received_at"])
    max_date = temp["received_at"].max()
    recent_start = max_date - pd.Timedelta(days=recent_days-1)
    baseline_start = recent_start - pd.Timedelta(days=baseline_days)

    recent = temp[temp["received_at"] >= recent_start].groupby(group_col).size()
    baseline = temp[
        (temp["received_at"] >= baseline_start) &
        (temp["received_at"] < recent_start)
    ].groupby(group_col).size()

    recent_daily = recent / recent_days
    baseline_daily = baseline / baseline_days

    result = pd.DataFrame({
        "recent_daily_avg": recent_daily,
        "baseline_daily_avg": baseline_daily
    }).fillna(0)

    result["increase_pct"] = (
        (result["recent_daily_avg"] - result["baseline_daily_avg"]) /
        result["baseline_daily_avg"].replace(0, 0.1) * 100
    )
    return result.sort_values("increase_pct", ascending=False).reset_index()
