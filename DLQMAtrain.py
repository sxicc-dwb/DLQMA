

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


SEED = 42
TRAIN_MODEL = True
ENABLE_PREDICTION = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "aug_thesis"
MODEL_DIR = PROJECT_ROOT / "model_thesis"
FINAL_MODEL_PATH = MODEL_DIR / "DLQMA_thesis_final.h5"
BEST_MODEL_PATH = MODEL_DIR / "DLQMA_thesis_best.h5"
TRAINING_LOG_PATH = MODEL_DIR / "training_log.csv"

# These values are intentionally identical to reallynewDeepMID.py.
BATCH_SIZE = 64
CONV_LAYERS = 6
PHASE1_EPOCHS = 80
PHASE2_EPOCHS = 20
PHASE3_EPOCHS = 30
LEARNING_RATE_REG = 0.0001

sys.path.insert(0, str(PROJECT_ROOT))

from reallynewDeepMID import (  # noqa: E402
    create_multitask_model,
    load_model,
    train_enhanced_model,
)


def set_random_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(path: Path) -> dict:
    import pickle

    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")
    with path.open("rb") as handle:
        dataset = pickle.load(handle)

    required = {"R", "S", "y", "conc"}
    missing = required.difference(dataset)
    if missing:
        raise KeyError(f"{path.name} 缺少字段: {sorted(missing)}")
    if dataset.get("conc_definition") != "target_ratio_of_reference_standard":
        raise ValueError(f"{path.name} 不是锁定的 target_ratio 标签数据")

    y = np.asarray(dataset["y"]).reshape(-1)
    conc = np.asarray(dataset["conc"]).reshape(-1)
    positive = conc[y == 1]
    negative = conc[y == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError(f"{path.name} 缺少正样本或负样本")
    if positive.min() < 0.2 - 1e-6 or positive.max() > 1.0 + 1e-6:
        raise ValueError(f"{path.name} 的正样本 conc 不在 0.2–1.0 范围内")
    if not np.allclose(negative, 0.0):
        raise ValueError(f"{path.name} 的负样本 conc 不全为 0")

    dataset["conc"] = conc.reshape(-1, 1)
    return dataset


def build_callbacks() -> list:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = MODEL_DIR / "tensorboard"
    return [
        EarlyStopping(monitor="val_loss", patience=20, min_delta=0.001),
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        CSVLogger(str(TRAINING_LOG_PATH), append=True, separator=","),
    ]


def train() -> tf.keras.Model:
    train_data = load_dataset(DATA_DIR / "dlqma_train.pkl")
    valid_data = load_dataset(DATA_DIR / "dlqma_valid.pkl")

    print(
        f"训练集 {len(train_data['y'])} 对；"
        f"验证集 {len(valid_data['y'])} 对"
    )
    model = create_multitask_model(
        [train_data["R"].shape, train_data["S"].shape],
        num_conv_layers=CONV_LAYERS,
        lr=LEARNING_RATE_REG,
    )
    train_enhanced_model(
        model,
        [train_data["R"], train_data["S"]],
        train_data["y"],
        train_data["conc"],
        batch_size=BATCH_SIZE,
        phase1_epochs=PHASE1_EPOCHS,
        phase2_epochs=PHASE2_EPOCHS,
        phase3_epochs=PHASE3_EPOCHS,
        Xs_valid=[valid_data["R"], valid_data["S"]],
        y_valid=valid_data["y"],
        conc_valid=valid_data["conc"],
        callbacks=build_callbacks(),
    )
    model.save(FINAL_MODEL_PATH)
    print(f"最终模型已保存到: {FINAL_MODEL_PATH}")
    print(f"最佳验证模型已保存到: {BEST_MODEL_PATH}")
    print(f"训练日志已保存到: {TRAINING_LOG_PATH}")
    return model


def main() -> None:
    set_random_seed()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_MODEL:
        train()
    elif ENABLE_PREDICTION:
        if not FINAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到最终模型: {FINAL_MODEL_PATH}")
        load_model(str(FINAL_MODEL_PATH.with_suffix("")))
        print(
            "模型加载成功。实验样品预测请运行 "
            "DLQMA_thesis_predict_mixtures.py。"
        )
    else:
        print("TRAIN_MODEL 和 ENABLE_PREDICTION 均为 False，未执行任务。")


if __name__ == "__main__":
    main()
