#!/usr/bin/env python3
"""
SWELL Analysis CORRECTED - Using REAL participants instead of synthetic ones
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def swell_analysis_real_participants():
    """Análisis CORRECTO de SWELL usando participantes REALES, no sintéticos."""

    print("🔍 SWELL ANALYSIS - PARTICIPANTES REALES vs SINTÉTICOS")
    print("=" * 60)

    # Cargar datos reales de computer interaction
    computer_file = "data/SWELL/3 - Feature dataset/per sensor/A - Computer interaction features (Ulog - All Features per minute)-Sheet_1.csv"

    df = pd.read_csv(computer_file)
    print(f"✓ Datos cargados: {df.shape}")
    print(f"  Participantes REALES: {sorted(df['PP'].unique())}")
    print(f"  Condiciones: {df['Condition'].value_counts().to_dict()}")
    print()

    # === MÉTODO 1: PARTICIPANTES REALES (25 participantes) ===
    print("🧑‍🔬 MÉTODO 1: Participantes REALES (PP1-PP25)")
    print("-" * 40)

    # Preparar datos
    features_cols = [
        "SnMouseAct",
        "SnLeftClicked",
        "SnRightClicked",
        "SnDoubleClicked",
        "SnWheel",
        "SnDragged",
        "SnMouseDistance",
        "SnKeyStrokes",
        "SnChars",
        "SnSpecialKeys",
        "SnDirectionKeys",
        "SnErrorKeys",
        "SnShortcutKeys",
        "SnSpaces",
        "SnAppChange",
        "SnTabfocusChange",
    ]

    # Limpiar datos - convertir todo a numérico
    X_real = df[features_cols].replace("#VALUE!", np.nan)
    X_real = X_real.apply(pd.to_numeric, errors="coerce").fillna(0)
    y_real = df["Condition"].isin(["T", "I", "R"]).astype(int)
    subjects_real = df["PP"]

    print(f"  Features: {X_real.shape[1]}")
    print(f"  Muestras: {len(X_real)}")
    print(f"  Participantes únicos: {len(subjects_real.unique())}")

    # División por participantes REALES
    unique_participants = subjects_real.unique()
    train_participants = unique_participants[: len(unique_participants) // 2]

    train_mask = subjects_real.isin(train_participants)

    X_train_real = X_real[train_mask]
    X_test_real = X_real[~train_mask]
    y_train_real = y_real[train_mask]
    y_test_real = y_real[~train_mask]

    print(f"  Train participants: {sorted(train_participants)}")
    print(
        f"  Test participants: {sorted([p for p in unique_participants if p not in train_participants])}"
    )
    print(f"  Train samples: {len(X_train_real)}, Test samples: {len(X_test_real)}")

    # Entrenar con participantes reales
    scaler_real = StandardScaler()
    X_train_real_scaled = scaler_real.fit_transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    rf_real = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_real.fit(X_train_real_scaled, y_train_real)

    y_pred_real = rf_real.predict(X_test_real_scaled)
    accuracy_real = accuracy_score(y_test_real, y_pred_real)

    print(f"  🎯 ACCURACY REAL: {accuracy_real:.3f} ({accuracy_real*100:.1f}%)")
    print()

    # === MÉTODO 2: PARTICIPANTES SINTÉTICOS (como el script original) ===
    print("🤖 MÉTODO 2: Participantes SINTÉTICOS (P01-P500)")
    print("-" * 40)

    # Simular lo que hace el script original - crear sujetos sintéticos
    synthetic_subjects = [f"P{(i//100) + 1:02d}" for i in range(len(df))]

    print(f"  Sujetos sintéticos creados: {len(set(synthetic_subjects))}")
    print(f"  Primeros 5: {list(set(synthetic_subjects))[:5]}")
    print(f"  Últimos 5: {list(set(synthetic_subjects))[-5:]}")

    # División por participantes sintéticos
    unique_synthetic = list(set(synthetic_subjects))
    train_synthetic = unique_synthetic[: len(unique_synthetic) // 2]

    train_mask_synthetic = pd.Series(synthetic_subjects).isin(train_synthetic)

    X_train_synthetic = X_real[train_mask_synthetic]
    X_test_synthetic = X_real[~train_mask_synthetic]
    y_train_synthetic = y_real[train_mask_synthetic]
    y_test_synthetic = y_real[~train_mask_synthetic]

    print(f"  Train synthetic subjects: {len(train_synthetic)} subjects")
    print(
        f"  Test synthetic subjects: {len(unique_synthetic) - len(train_synthetic)} subjects"
    )
    print(
        f"  Train samples: {len(X_train_synthetic)}, Test samples: {len(X_test_synthetic)}"
    )

    # Entrenar con participantes sintéticos
    scaler_synthetic = StandardScaler()
    X_train_synthetic_scaled = scaler_synthetic.fit_transform(X_train_synthetic)
    X_test_synthetic_scaled = scaler_synthetic.transform(X_test_synthetic)

    rf_synthetic = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_synthetic.fit(X_train_synthetic_scaled, y_train_synthetic)

    y_pred_synthetic = rf_synthetic.predict(X_test_synthetic_scaled)
    accuracy_synthetic = accuracy_score(y_test_synthetic, y_pred_synthetic)

    print(
        f"  🎯 ACCURACY SINTÉTICO: {accuracy_synthetic:.3f} ({accuracy_synthetic*100:.1f}%)"
    )
    print()

    # === COMPARACIÓN Y DIAGNÓSTICO ===
    print("📊 COMPARACIÓN FINAL")
    print("-" * 20)
    print(
        f"🧑‍🔬 Participantes REALES (25):    {accuracy_real:.3f} ({accuracy_real*100:.1f}%)"
    )
    print(
        f"🤖 Participantes SINTÉTICOS (500): {accuracy_synthetic:.3f} ({accuracy_synthetic*100:.1f}%)"
    )
    print()

    if accuracy_synthetic > accuracy_real + 0.05:
        print(
            "🚨 DIAGNÓSTICO: Los participantes sintéticos inflan artificialmente el accuracy"
        )
        print("   Razones:")
        print("   1. Cada 'sujeto sintético' tiene solo ~100 muestras consecutivas")
        print("   2. No hay variabilidad real entre 'sujetos'")
        print("   3. División temporal no es lo mismo que división por sujetos")
        print(f"   4. El {accuracy_synthetic*100:.1f}% es ARTIFICIAL")
    else:
        print("✅ DIAGNÓSTICO: Ambos métodos dan resultados similares")
        print("   Los datos SWELL son genuinamente buenos")

    print()
    print("💡 CONCLUSIÓN:")
    print(f"   - Accuracy REAL con participantes reales: {accuracy_real*100:.1f}%")
    print(f"   - Accuracy FALSO con sujetos sintéticos: {accuracy_synthetic*100:.1f}%")
    print(
        f"   - Diferencia: {abs(accuracy_synthetic - accuracy_real)*100:.1f} puntos porcentuales"
    )

    if accuracy_synthetic > 0.95:
        print("   - El 99.6% del script original es SOSPECHOSO por sujetos sintéticos")
    else:
        print("   - Los resultados son consistentes entre métodos")


if __name__ == "__main__":
    swell_analysis_real_participants()
