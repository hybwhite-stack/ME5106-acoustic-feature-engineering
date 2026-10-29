"""Select power features using held-out known groups, then predict unknown P.

Power uses the 46 SampleIDs with known power. Speed exploration uses all 47
SampleIDs with known speed, including Sample 49. Its speed is held out only
within that group's cross-validation fold. Unknown power labels never enter
training, scaling, or feature selection.
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
EARLY_RMS_MEDIAN = "前0至10ms对中10至25ms_RMS比_五轨中位数"
EARLY_HIGH_MEDIAN = "前0至10ms减中10至25ms_高频比例_五轨中位数"
TAIL = ("尾25ms至终点对中10至25ms_RMS比_五轨中位数",
        "尾25ms至终点减中10至25ms_高频比例_五轨中位数")
DURATION = ("Duration_s_Mean", "Duration_s_SD")

OLD_EIGHT_MEDIAN = BASE_FIVE + (ONSET, EARLY_RMS_MEDIAN, EARLY_HIGH_MEDIAN)
POWER_SETS = {
    "A_六项时域": SIX,
    "B_五项声强频谱": BASE_FIVE,
    "C_八项加入尖峰与分段": OLD_EIGHT_MEDIAN,
    "D_十一项时域声强频谱": SIX + BASE_FIVE,
    "E_十四项全部联合": SIX + OLD_EIGHT_MEDIAN,
}
SPEED_FEATURES = SIX + BASE_FIVE + (ONSET, EARLY_RMS_MEDIAN,
                                     EARLY_HIGH_MEDIAN) + DURATION + TAIL
SPEED_SETS = {
    "V1_六项时域": SIX,
    "V2_五项声强频谱": BASE_FIVE,
    "V3_八项加入尖峰与前中变化": BASE_FIVE + (
        ONSET, EARLY_RMS_MEDIAN, EARLY_HIGH_MEDIAN),
    "V4_十四项联合": SIX + BASE_FIVE + (
        ONSET, EARLY_RMS_MEDIAN, EARLY_HIGH_MEDIAN),
    "V5_十六项加时长": SPEED_FEATURES[:-2],
    "V6_十八项再加尾段": SPEED_FEATURES,
}


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
    joined = []
    for sid in range(1, 50):
        row = dict(base[sid])
        for feature in SIX:
            row[feature] = float(six[sid][feature])
        row[ONSET] = float(onset[sid][ONSET])
        for feature in (EARLY_RMS_MEDIAN, EARLY_HIGH_MEDIAN) + TAIL:
            row[feature] = float(windows[sid][feature])
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
    power_known = [row for row in rows if row["PowerKnown"] == "1"]
    power_unknown = [row for row in rows if row["PowerKnown"] != "1"]
    speed_known = [row for row in rows if row["SpeedKnown"] == "1"]
    speed_unknown = [row for row in rows if row["SpeedKnown"] != "1"]
    assert len(power_known) == 46 and len(speed_known) == 47
    assert {int(row["SampleID"]) for row in power_unknown} == {41, 43, 49}
    assert {int(row["SampleID"]) for row in speed_unknown} == {41, 43}
    assert all(row["LaserPower_W"].lower() == "nan" for row in power_unknown)
    assert int(float(next(row for row in speed_known if row["SampleID"] == "49")
                     ["ScanSpeed_mm_s"])) == 850

    power_group_ids = np.array([int(row["SampleID"]) for row in power_known])
    power = np.array([float(row["LaserPower_W"]) for row in power_known])
    speed_group_ids = np.array([int(row["SampleID"]) for row in speed_known])
    speed = np.array([int(float(row["ScanSpeed_mm_s"])) for row in speed_known])
    power_summary, power_details = [], []

    for set_name, feature_names in POWER_SETS.items():
        x = matrix(power_known, feature_names)
        loo = np.empty(46)
        for i in range(46):
            keep = np.arange(46) != i
            model = power_model().fit(x[keep], power[keep])
            loo[i] = float(model.predict(x[i:i + 1])[0])
            power_details.append({
                "特征组合": set_name, "SampleID": int(power_group_ids[i]),
                "真实功率_W": float(power[i]), "留出预测功率_W": float(loo[i]),
                "相对误差_pct": float(100 * abs(loo[i] - power[i]) / power[i]),
            })
        relative = np.abs(loo - power) / power
        success_count = int(np.sum(relative <= 0.20))
        power_summary.append({
            "特征组合": set_name, "特征数": len(feature_names),
            "具体特征列名": "；".join(feature_names),
            "固定模型": "StandardScaler + ElasticNet(alpha=0.1,l1_ratio=0.3)",
            "46组误差不超过20pct": success_count,
            "46组成功率_pct": float(100 * success_count / len(power)),
            "46组MAPE_pct": float(100 * np.mean(relative)),
        })

    # Select only from held-out predictions on known-power SampleIDs.
    selected = min(power_summary, key=lambda row: (
        -row["46组误差不超过20pct"], row["46组MAPE_pct"], row["特征数"]))
    selected_name = selected["特征组合"]
    selected_features = POWER_SETS[selected_name]
    final_power = power_model().fit(matrix(power_known, selected_features), power)
    final_predictions = final_power.predict(matrix(power_unknown, selected_features))
    unknown_power = {int(row["SampleID"]): float(pred)
                     for row, pred in zip(power_unknown, final_predictions)}

    speed_summary, speed_details = [], []
    for set_name, feature_names in SPEED_SETS.items():
        speed_x = matrix(speed_known, feature_names)
        speed_loo = np.empty(47, dtype=int)
        for i in range(47):
            keep = np.arange(47) != i
            model = speed_model().fit(speed_x[keep], speed[keep])
            speed_loo[i] = int(model.predict(speed_x[i:i + 1])[0])
            speed_details.append({
                "特征组合": set_name, "SampleID": int(speed_group_ids[i]),
                "真实速度_mm_s": int(speed[i]),
                "留出预测速度_mm_s": int(speed_loo[i]),
                "正确": int(speed_loo[i] == speed[i]),
            })
        pred49_held_out = int(speed_loo[np.where(speed_group_ids == 49)[0][0]])
        class_correct = {value: int(np.sum((speed == value) & (speed_loo == value)))
                         for value in (550, 700, 850)}
        balanced = np.mean([class_correct[value] / np.sum(speed == value)
                            for value in (550, 700, 850)])
        speed_summary.append({
            "特征组合": set_name, "特征数": len(feature_names),
            "具体特征列名": "；".join(feature_names),
            "固定模型": "StandardScaler + LogisticRegression(C=0.3)",
            "已知速度训练组数": 47,
            "47组正确": int(np.sum(speed_loo == speed)),
            "47组准确率_pct": float(100 * np.mean(speed_loo == speed)),
            "47组平衡准确率_pct": float(100 * balanced),
            "550正确_总12": class_correct[550],
            "700正确_总21": class_correct[700],
            "850正确_总14": class_correct[850],
            "49留一预测速度_mm_s": pred49_held_out,
            "49留一是否正确": int(pred49_held_out == 850),
        })
    selected_speed = min(speed_summary, key=lambda row: (
        -row["47组正确"], -row["47组平衡准确率_pct"], row["特征数"]))
    selected_speed_name = selected_speed["特征组合"]
    final_speed = speed_model().fit(
        matrix(speed_known, SPEED_SETS[selected_speed_name]), speed)
    assert final_speed.classes_.tolist() == [550, 700, 850]
    speed_predictions = final_speed.predict(
        matrix(speed_unknown, SPEED_SETS[selected_speed_name]))
    unknown_speed = {int(row["SampleID"]): int(pred)
                     for row, pred in zip(speed_unknown, speed_predictions)}
    predictions = []
    for sid in (41, 43, 49):
        predictions.append({
            "SampleID": sid,
            "功率特征组合": selected_name,
            "功率模型": "StandardScaler + ElasticNet(alpha=0.1,l1_ratio=0.3)",
            "预测功率_W": unknown_power[sid],
            "速度特征组合": selected_speed_name if sid != 49 else "",
            "速度模型": "StandardScaler + LogisticRegression(C=0.3)" if sid != 49 else "",
            "速度_mm_s": 850 if sid == 49 else unknown_speed[sid],
            "速度状态": "已知" if sid == 49 else "预测",
        })
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv("固定ElasticNet特征消融.csv", power_summary)
    write_csv("46组功率逐组留出.csv", power_details)
    write_csv("固定逻辑回归速度特征对照.csv", speed_summary)
    write_csv("47组速度逐组留出.csv", speed_details)
    write_csv("三个待预测组_统一方案.csv", predictions)
    print("Fixed-model feature comparison:")
    for result in power_summary:
        print(result)
    print("Selected power features:", selected_name,
          "held-out success", selected["46组误差不超过20pct"], "/46",
          "MAPE", round(selected["46组MAPE_pct"], 2), "%")
    print("Fixed speed-model feature comparison:")
    for result in speed_summary:
        print(result["特征组合"], result["47组正确"], "/47",
              "balanced accuracy", round(result["47组平衡准确率_pct"], 2),
              "49 held-out", result["49留一预测速度_mm_s"])
    print("Selected speed features:", selected_speed_name,
          "held-out correct", selected_speed["47组正确"], "/47",
          "balanced accuracy", round(selected_speed["47组平衡准确率_pct"], 2), "%")
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()

