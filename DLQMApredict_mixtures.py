

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model_thesis" / "DLQMA_thesis_final.h5"
STANDARDS_DIR = PROJECT_ROOT / "mydata" / "standards"
MIXTURES_DIR = PROJECT_ROOT / "mydata" / "mixture"
RESULTS_DIR = PROJECT_ROOT / "results_thesis"
PER_SAMPLE_DIR = RESULTS_DIR / "experimental_mixtures"
FINAL_RESULTS_PATH = RESULTS_DIR / "experimental_results_thesis.csv"

# Optional manually curated file. See create_truth_template().
TRUTH_PATH = Path(__file__).resolve().parent / "experimental_truth.csv"
TRUTH_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "experimental_truth_template.csv"
)

CLASSIFICATION_THRESHOLD = 0.5

sys.path.insert(0, str(PROJECT_ROOT))
from readBruker import read_bruker_hs_base  # noqa: E402
from reallynewDeepMID import SpatialPyramidPooling  # noqa: E402


TRUTH_COLUMNS = [
    "Sample",
    "Component",
    "Actually_present",
    "Theoretical_relative_value",
    "Remark",
]
OUTPUT_COLUMNS = [
    "Sample",
    "Component",
    "Actually_present",
    "Probability",
    "Predicted_relative_value",
    "Theoretical_relative_value",
    "Absolute_error",
    "Relative_error",
    "Remark",
]


def validate_paths() -> None:
    required = [MODEL_PATH, STANDARDS_DIR, MIXTURES_DIR]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少以下输入:\n" + "\n".join(missing))


def expected_spectral_points(model) -> int:
    input_shapes = model.input_shape
    if not isinstance(input_shapes, list) or len(input_shapes) != 2:
        raise ValueError(f"模型应有两个输入，实际 input_shape={input_shapes}")
    lengths = {shape[1] for shape in input_shapes}
    if len(lengths) != 1 or None in lengths:
        raise ValueError(f"无法确定模型要求的光谱长度: {input_shapes}")
    return int(next(iter(lengths)))


def validate_spectrum(spectrum: dict, expected_points: int, kind: str) -> None:
    fid = np.asarray(spectrum["fid"]).reshape(-1)
    if fid.size != expected_points:
        raise ValueError(
            f"{kind}“{spectrum['name']}”有 {fid.size} 个点，"
            f"模型要求 {expected_points} 个点"
        )
    if not np.all(np.isfinite(fid)):
        raise ValueError(f"{kind}“{spectrum['name']}”包含 NaN 或无穷值")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")
    return cleaned or "unnamed_sample"


def load_truth() -> pd.DataFrame | None:
    if not TRUTH_PATH.exists():
        return None
    truth = pd.read_csv(TRUTH_PATH, encoding="utf-8-sig")
    missing = set(TRUTH_COLUMNS).difference(truth.columns)
    if missing:
        raise KeyError(f"{TRUTH_PATH.name} 缺少字段: {sorted(missing)}")
    if truth.duplicated(["Sample", "Component"]).any():
        duplicates = truth.loc[
            truth.duplicated(["Sample", "Component"], keep=False),
            ["Sample", "Component"],
        ]
        raise ValueError(
            "experimental_truth.csv 存在重复的样品-组分记录:\n"
            + duplicates.to_string(index=False)
        )
    return truth[TRUTH_COLUMNS]


def predict_one_mixture(
    model,
    standards: list[dict],
    mixture: dict,
    expected_points: int,
) -> pd.DataFrame:
    validate_spectrum(mixture, expected_points, "混合谱")
    reference = np.stack(
        [np.asarray(item["fid"], dtype=np.float32) for item in standards]
    )
    query = np.repeat(
        np.asarray(mixture["fid"], dtype=np.float32)[np.newaxis, :],
        repeats=len(standards),
        axis=0,
    )
    probabilities, relative_values = model.predict(
        [reference[..., np.newaxis], query[..., np.newaxis]],
        verbose=0,
    )
    return pd.DataFrame(
        {
            "Sample": mixture["name"],
            "Component": [item["name"] for item in standards],
            "Probability": np.asarray(probabilities).reshape(-1),
            "Predicted_relative_value": np.asarray(relative_values).reshape(-1),
        }
    )


def attach_truth(
    predictions: pd.DataFrame,
    truth: pd.DataFrame | None,
) -> pd.DataFrame:
    if truth is None:
        result = predictions.copy()
        result["Actually_present"] = pd.NA
        result["Theoretical_relative_value"] = np.nan
        result["Remark"] = ""
    else:
        result = predictions.merge(
            truth,
            on=["Sample", "Component"],
            how="left",
            validate="one_to_one",
        )

    theoretical = pd.to_numeric(
        result["Theoretical_relative_value"],
        errors="coerce",
    )
    predicted = result["Predicted_relative_value"].astype(float)
    has_theoretical = theoretical.notna()
    result["Absolute_error"] = np.where(
        has_theoretical,
        np.abs(predicted - theoretical),
        np.nan,
    )
    result["Relative_error"] = np.where(
        has_theoretical & theoretical.ne(0),
        result["Absolute_error"] / np.abs(theoretical),
        np.nan,
    )
    result["Theoretical_relative_value"] = theoretical
    result["Remark"] = result["Remark"].fillna("")
    return result[OUTPUT_COLUMNS]


def create_truth_template(predictions: pd.DataFrame) -> None:
    template = predictions[["Sample", "Component"]].copy()
    template["Actually_present"] = ""
    template["Theoretical_relative_value"] = ""
    template["Remark"] = ""
    template.to_csv(
        TRUTH_TEMPLATE_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"未找到 {TRUTH_PATH.name}；已生成待填写模板: "
        f"{TRUTH_TEMPLATE_PATH}"
    )
    print("没有严格理论值的单元格请保持为空，不会计算误差。")


def print_summary(result: pd.DataFrame) -> None:
    for sample, rows in result.groupby("Sample", sort=False):
        detected = rows[rows["Probability"] >= CLASSIFICATION_THRESHOLD]
        print(f"\n{sample}: 检出 {len(detected)}/{len(rows)} 个候选组分")
        if detected.empty:
            print("  未检出组分")
            continue
        display = detected[
            ["Component", "Probability", "Predicted_relative_value"]
        ].sort_values("Probability", ascending=False)
        print(display.to_string(index=False))


def main() -> None:
    validate_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(
        MODEL_PATH,
        custom_objects={"SpatialPyramidPooling": SpatialPyramidPooling},
        compile=False,
    )
    expected_points = expected_spectral_points(model)

    print(f"读取标准谱: {STANDARDS_DIR}")
    standards = read_bruker_hs_base(
        str(STANDARDS_DIR),
        False,
        True,
        False,
    )
    print(f"读取实验混合谱: {MIXTURES_DIR}")
    mixtures = read_bruker_hs_base(
        str(MIXTURES_DIR),
        False,
        True,
        False,
    )
    if not standards:
        raise RuntimeError("没有读取到标准谱")
    if not mixtures:
        raise RuntimeError("没有读取到实验混合谱")
    for standard in standards:
        validate_spectrum(standard, expected_points, "标准谱")

    prediction_tables = [
        predict_one_mixture(model, standards, mixture, expected_points)
        for mixture in mixtures
    ]
    predictions = pd.concat(prediction_tables, ignore_index=True)
    truth = load_truth()
    result = attach_truth(predictions, truth)

    result.to_csv(
        FINAL_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    for sample, rows in result.groupby("Sample", sort=False):
        rows.to_csv(
            PER_SAMPLE_DIR / f"{safe_filename(sample)}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if truth is None:
        create_truth_template(predictions)
    print_summary(result)
    print(f"\n统一结果已保存到: {FINAL_RESULTS_PATH}")
    print(f"各样品结果已保存到: {PER_SAMPLE_DIR}")


if __name__ == "__main__":
    main()
