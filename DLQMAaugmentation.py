

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "aug_thesis"
CHECK_PATH = PROJECT_ROOT / "DLQMA_thesis_dataset_check.csv"

TRAIN_POSITIVE_COUNT = 50_000
VALID_POSITIVE_COUNT = 5_000
TEST_POSITIVE_COUNT = 5_000
MAX_COMPONENTS = 5
NOISE_LEVEL = 0.0001
CONCENTRATION_COMPONENTS = 1
LOW_RATIO = 0.2
HIGH_RATIO = 1.0

sys.path.insert(0, str(PROJECT_ROOT))

from readBruker import read_bruker_hs_base  # noqa: E402
from reallaug import data_augmentation, save_data  # noqa: E402


def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_dataset(spectra: list, positive_count: int) -> dict:
    return data_augmentation(
        spectra=spectra,
        n=positive_count,
        max_pc=MAX_COMPONENTS,
        noise_level=NOISE_LEVEL,
        concentration_components=CONCENTRATION_COMPONENTS,
        low=LOW_RATIO,
        high=HIGH_RATIO,
    )


def check_dataset(name: str, dataset: dict) -> dict:
    y = np.asarray(dataset["y"]).reshape(-1)
    conc = np.asarray(dataset["conc"]).reshape(-1)
    positive = conc[y == 1]
    negative = conc[y == 0]

    if len(y) != len(conc):
        raise ValueError(f"{name}: y and conc lengths differ")
    if positive.size == 0 or negative.size == 0:
        raise ValueError(f"{name}: positive or negative samples are missing")

    row = {
        "Dataset": name,
        "N_total": int(y.size),
        "N_positive": int(positive.size),
        "N_negative": int(negative.size),
        "Positive_conc_min": float(positive.min()),
        "Positive_conc_max": float(positive.max()),
        "Positive_conc_mean": float(positive.mean()),
        "Positive_conc_gt1_fraction": float(np.mean(positive > 1.0)),
        "Negative_conc_max": float(negative.max()),
    }

    tolerance = 1e-6
    if row["Positive_conc_min"] < LOW_RATIO - tolerance:
        raise ValueError(f"{name}: positive conc is below {LOW_RATIO}")
    if row["Positive_conc_max"] > HIGH_RATIO + tolerance:
        raise ValueError(f"{name}: positive conc is above {HIGH_RATIO}")
    if row["Positive_conc_gt1_fraction"] != 0.0:
        raise ValueError(f"{name}: positive conc contains values above 1")
    if not np.allclose(negative, 0.0):
        raise ValueError(f"{name}: negative conc is not entirely zero")
    if dataset.get("conc_definition") != "target_ratio_of_reference_standard":
        raise ValueError(f"{name}: conc_definition is missing or incorrect")

    return row


def save_checks(rows: list[dict]) -> None:
    with CHECK_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_random_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    standards_path = PROJECT_ROOT / "mydata" / "standards"
    print(f"读取标准光谱: {standards_path}")
    spectra = read_bruker_hs_base(str(standards_path), False, True, False)
    if len(spectra) < 2:
        raise RuntimeError("至少需要两个标准光谱才能生成正负样本")
    print(f"读取了 {len(spectra)} 个标准光谱")

    specifications = [
        ("Train", TRAIN_POSITIVE_COUNT, OUTPUT_DIR / "dlqma_train.pkl"),
        ("Valid", VALID_POSITIVE_COUNT, OUTPUT_DIR / "dlqma_valid.pkl"),
        ("Test", TEST_POSITIVE_COUNT, OUTPUT_DIR / "dlqma_test.pkl"),
    ]

    check_rows = []
    for name, positive_count, output_path in specifications:
        print(f"\n生成 {name} 数据集（正负样本各 {positive_count}）...")
        dataset = generate_dataset(spectra, positive_count)
        check_rows.append(check_dataset(name, dataset))
        save_data(dataset, str(output_path))

    save_checks(check_rows)
    print(f"\n标签检查通过，检查结果已保存到: {CHECK_PATH}")
    for row in check_rows:
        print(row)


if __name__ == "__main__":
    main()
