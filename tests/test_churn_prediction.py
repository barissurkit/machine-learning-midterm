import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_COLUMNS = [
    "yas",
    "aylik_gelir",
    "abonelik_suresi_ay",
    "destek_talebi_sayisi",
    "sehir",
    "uyelik_tipi",
    "aylik_ucret",
    "son_giris_gun_once",
    "otomatik_odeme",
    "churn",
]


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(ROOT / "musteri_churn.csv")


@pytest.fixture(scope="module")
def script_output():
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "churn_prediction.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_dataset_has_expected_columns(dataset):
    assert list(dataset.columns) == EXPECTED_COLUMNS


def test_dataset_has_200_rows(dataset):
    assert len(dataset) == 200


def test_target_is_binary(dataset):
    assert set(dataset["churn"].unique()) == {0, 1}


def test_dataset_has_missing_values_only_in_documented_columns(dataset):
    missing = dataset.columns[dataset.isna().any()].tolist()
    assert set(missing) <= {"aylik_gelir", "sehir", "son_giris_gun_once"}


def test_script_reports_the_split_sizes(script_output):
    assert "Train veri sayısı: 140" in script_output
    assert "Validation veri sayısı: 30" in script_output
    assert "Test veri sayısı: 30" in script_output


def test_script_compares_both_models_on_validation(script_output):
    assert "Logistic Regression" in script_output
    assert "KNN" in script_output
    assert re.search(r"en iyi validation sonucuna sahip model: (Logistic Regression|KNN)", script_output)


def test_script_prints_test_metrics_between_zero_and_one(script_output):
    test_part = script_output.split("TEST SONUÇLARI", 1)[1]
    for metric in ("Accuracy", "Precision", "Recall", "F1-score"):
        match = re.search(rf"{metric}\s*: (\d\.\d{{4}})", test_part)
        assert match, f"{metric} çıktıda bulunamadı"
        assert 0.0 <= float(match.group(1)) <= 1.0


def test_script_prints_confusion_matrix_with_all_test_samples(script_output):
    test_part = script_output.split("Confusion Matrix:", 1)[1]
    numbers = re.findall(r"\d+", test_part.split("Classification Report", 1)[0])
    assert len(numbers) == 4
    assert sum(int(n) for n in numbers) == 30
