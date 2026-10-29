"""Small-data, group-level acoustic feature comparison and P/V prediction.

Only 46 fully labeled SampleIDs fit models. Each SampleID has already been
summarized from five tracks. Sample 49's 425 W course-slide reference is used
solely after prediction, for an explicitly retrospective error calculation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
RESULTS = ROOT / "results"

SIX = ("Min_Mean", "Max_Mean", "Mean_Mean", "Median_Mean",
       "PeakToPeak_Mean", "RMS_Mean")
BASE_FIVE = ("PooledRMS", "PeakAbs_Mean", "SpectralCentroid_Hz_Mean",
             "Band8to20k_Fraction_Mean", "Band40to80k_Fraction_Mean")
ONSET = "起始尖峰比_最突出两条均值"
EARLY_RMS_MEAN = "前0至10ms对中10至25ms_RMS比_五轨平均数"
EARLY_HIGH_MEAN = "前0至10ms减中10至25ms_高频比例_五轨平均数"
EARLY_RMS_MEDIAN = "前0至10ms对中10至25ms_RMS比_五轨中位数"
EARLY_HIGH_MEDIAN = "前0至10ms减中10至25ms_高频比例_五轨中位数"
TAIL = ("尾25ms至终点对中10至25ms_RMS比_五轨中位数",
        "尾25ms至终点减中10至25ms_高频比例_五轨中位数")
DURATION = ("Duration_s_Mean", "Duration_s_SD")

OLD_EIGHT_MEAN = BASE_FIVE + (ONSET, EARLY_RMS_MEAN, EARLY_HIGH_MEAN)
POWER_SETS = {
    "A_六项时域": SIX,
    "B_五项声强频谱": BASE_FIVE,
    "C_八项加入尖峰与分段": OLD_EIGHT_MEAN,
    "D_十一项时域声强频谱": SIX + BASE_FIVE,
    "E_十四项全部联合": SIX + OLD_EIGHT_MEAN,
}
SPEED_FEATURES = SIX + BASE_FIVE + (ONSET, EARLY_RMS_MEDIAN,
                                     EARLY_HIGH_MEDIAN) + DURATION + TAIL


def load_csv(name: str) -> list[dict[str, str]]:
    with (FEATURES / name).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def keyed(name: str, key: str) -> dict[int, dict[str, str]]:
    rows = load_csv(name)
    result = {int(row[key]): row for row in rows}
    assert len(rows) == len(result) == 49
    return result


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (RESULTS / name).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assemble() -> list[dict]:
    base = keyed("特征表.csv", "SampleID")
    six = keyed("六项时域统计_逐组49条.csv", "SampleID")
    onset = keyed("每组起始尖峰汇总.csv", "样本编号")
    windows = keyed("每组两三段特征.csv", "SampleID")
    per_track = load_csv("每条轨迹两三段特征.csv")
    assert len(per_track) == 245
    tracks: dict[int, list[dict[str, str]]] = {sid: [] for sid in range(1, 50)}
    for row in per_track:
        tracks[int(row["SampleID"])].append(row)
    assert all(len(five) == 5 for five in tracks.values())

    joined = []
    for sid in range(1, 50):
        row = dict(base[sid])
        for feature in SIX:
            row[feature] = float(six[sid][feature])
        row[ONSET] = float(onset[sid][ONSET])
        for feature in (EARLY_RMS_MEDIAN, EARLY_HIGH_MEDIAN) + TAIL:
            row[feature] = float(windows[sid][feature])
        row[EARLY_RMS_MEAN] = float(np.mean(
            [float(track["前中RMS比"]) for track in tracks[sid]]))
        row[EARLY_HIGH_MEAN] = float(np.mean(
            [float(track["前减中高频比例"]) for track in tracks[sid]]))
        joined.append(row)
    return joined


def power_model():
    return make_pipeline(StandardScaler(), ElasticNet(
        alpha=0.1, l1_ratio=0.3, max_iter=20_000))


def speed_model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.3, max_iter=10_000))


def matrix(rows: list[dict], features: tuple[str, ...]) -> np.ndarray:
    x = np.array([[float(row[f]) for f in features] for row in rows])
    assert np.isfinite(x).all()
    return x


def main() -> None:
    rows = assemble()
    known = [row for row in rows if row["FullyLabeled"] == "1"]
    unknown = [row for row in rows if row["FullyLabeled"] != "1"]
    assert len(known) == 46
    assert {int(row["SampleID"]) for row in unknown} == {41, 43, 49}
    assert all(row["LaserPower_W"].lower() == "nan" for row in unknown)
    assert int(float(next(row for row in unknown if row["SampleID"] == "49")
                     ["ScanSpeed_mm_s"])) == 850

    group_ids = np.array([int(row["SampleID"]) for row in known])
    power = np.array([float(row["LaserPower_W"]) for row in known])
    speed = np.array([int(float(row["ScanSpeed_mm_s"])) for row in known])
    power_summary, power_details = [], []
    unknown_power: dict[int, float] = {}

    for set_name, feature_names in POWER_SETS.items():
        x = matrix(known, feature_names)
        x_unknown = matrix(unknown, feature_names)
        loo = np.empty(46)
        for i in range(46):
            keep = np.arange(46) != i
            model = power_model().fit(x[keep], power[keep])
            loo[i] = float(model.predict(x[i:i + 1])[0])
            power_details.append({
                "特征组合": set_name, "SampleID": int(group_ids[i]),
                "真实功率_W": float(power[i]), "留出预测功率_W": float(loo[i]),
                "相对误差_pct": float(100 * abs(loo[i] - power[i]) / power[i]),
            })
        final = power_model().fit(x, power)
        predicted = np.asarray(final.predict(x_unknown)).reshape(-1)
        relative = np.abs(loo - power) / power
        pred49 = float(predicted[[int(r["SampleID"]) for r in unknown].index(49)])
        power_summary.append({
            "特征组合": set_name, "特征数": len(feature_names),
            "具体特征列名": "；".join(feature_names),
            "固定模型": "StandardScaler + ElasticNet(alpha=0.1,l1_ratio=0.3)",
            "46组误差不超过20pct": int(np.sum(relative <= 0.20)),
            "46组MAPE_pct": float(100 * np.mean(relative)),
            "49预测功率_W": pred49,
            "49相对425W事后误差_pct": float(100 * abs(pred49 - 425) / 425),
        })
        if set_name == "E_十四项全部联合":
            unknown_power = {int(row["SampleID"]): float(pred)
                             for row, pred in zip(unknown, predicted)}

    speed_x = matrix(known, SPEED_FEATURES)
    speed_unknown_x = matrix(unknown, SPEED_FEATURES)
    speed_loo = np.empty(46, dtype=int)
    for i in range(46):
        keep = np.arange(46) != i
        model = speed_model().fit(speed_x[keep], speed[keep])
        speed_loo[i] = int(model.predict(speed_x[i:i + 1])[0])
    speed_final = speed_model().fit(speed_x, speed)
    unknown_speed = {int(row["SampleID"]): int(pred)
                     for row, pred in zip(unknown,
                                          speed_final.predict(speed_unknown_x))}
    speed_details = [
        {"SampleID": int(sid), "真实速度_mm_s": int(truth),
         "留出预测速度_mm_s": int(pred), "正确": int(pred == truth)}
        for sid, truth, pred in zip(group_ids, speed, speed_loo)
    ]
    predictions = [
        {"SampleID": sid, "预测功率_W": unknown_power[sid],
         "预测速度_mm_s": unknown_speed[sid],
         "功率标签状态": "未用于训练",
         "备注": "49号425W仅课程材料事后参考" if sid == 49 else "正式真值待公布"}
        for sid in (41, 43, 49)
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv("固定ElasticNet特征消融.csv", power_summary)
    write_csv("46组功率逐组留出.csv", power_details)
    write_csv("46组速度逐组留出.csv", speed_details)
    write_csv("三个待预测组_统一方案.csv", predictions)
    print("Fixed-model feature comparison:")
    for result in power_summary:
        print(result)
    print("Speed leave-one-group-out correct:", int(np.sum(speed_loo == speed)), "/46")
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
