#!/usr/bin/env python3
"""
Análisis de la Pipeline de Procesamiento del Dataset SWELL
===========================================================
Este script analiza la relación entre los datos RAW (RRI) y los datos procesados (train/test)
"""

from pathlib import Path

import numpy as np
import pandas as pd


def analyze_raw_data():
    """Analiza los archivos RRI raw"""
    print("=" * 80)
    print("ANÁLISIS DE DATOS RAW (RRI - Intervalos R-R)")
    print("=" * 80)

    raw_dir = Path("data/SWELL/data/raw/rri")
    rri_files = sorted(raw_dir.glob("p*.txt"))

    print(f"\n📁 Archivos RRI encontrados: {len(rri_files)}")
    print(f"   Sujetos: {[f.stem for f in rri_files]}")

    # Analizar cada archivo
    stats = []
    for rri_file in rri_files:
        data = np.loadtxt(rri_file)
        subject_id = rri_file.stem
        stats.append(
            {
                "subject": subject_id,
                "num_samples": data.shape[0],
                "duration_sec": data[-1, 0] if len(data) > 0 else 0,
                "mean_rri": data[:, 1].mean(),
                "std_rri": data[:, 1].std(),
            }
        )

    df_stats = pd.DataFrame(stats)
    print("\n📊 Estadísticas por sujeto (RAW):")
    print(df_stats.to_string(index=False))

    return df_stats


def analyze_processed_data():
    """Analiza los datos procesados en train/test"""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE DATOS PROCESADOS (HRV Features)")
    print("=" * 80)

    df_train = pd.read_csv("data/SWELL/data/final/train.csv")
    df_test = pd.read_csv("data/SWELL/data/final/test.csv")

    print("\n📈 Dataset procesado:")
    print(f"   Train: {len(df_train):,} muestras")
    print(f"   Test:  {len(df_test):,} muestras")
    print(f"   Total: {len(df_train) + len(df_test):,} muestras")

    print(f"\n👥 Sujetos en train: {df_train['datasetId'].unique()}")
    print(f"   Sujetos en test:  {df_test['datasetId'].unique()}")

    print(f"\n🔢 Features extraídas: {len(df_train.columns) - 2}")

    # Categorizar features
    time_domain = [
        "MEAN_RR",
        "MEDIAN_RR",
        "SDRR",
        "RMSSD",
        "SDSD",
        "SDRR_RMSSD",
        "HR",
        "pNN25",
        "pNN50",
        "SD1",
        "SD2",
        "KURT",
        "SKEW",
    ]

    freq_domain = [
        "VLF",
        "VLF_PCT",
        "LF",
        "LF_PCT",
        "LF_NU",
        "HF",
        "HF_PCT",
        "HF_NU",
        "TP",
        "LF_HF",
        "HF_LF",
    ]

    nonlinear = ["sampen", "higuci"]

    relative = [c for c in df_train.columns if "REL_RR" in c]

    print("\n📊 Tipos de features:")
    print(
        f"   ⏱️  Dominio temporal: {len(time_domain)} ({', '.join(time_domain[:5])}...)"
    )
    print(
        f"   📈 Dominio frecuencia: {len(freq_domain)} ({', '.join(freq_domain[:5])}...)"
    )
    print(f"   🔀 No lineales: {len(nonlinear)} ({', '.join(nonlinear)})")
    print(f"   📐 Relativas: {len(relative)} ({len(relative)} features)")

    print("\n🎯 Distribución de clases:")
    combined = pd.concat([df_train, df_test])
    for condition, count in combined["condition"].value_counts().items():
        pct = count / len(combined) * 100
        print(f"   {condition:15s}: {count:6,} ({pct:5.1f}%)")

    return df_train, df_test


def explain_pipeline():
    """Explica la pipeline de procesamiento"""
    print("\n" + "=" * 80)
    print("PIPELINE DE PROCESAMIENTO")
    print("=" * 80)

    print(
        """
📝 TRANSFORMACIÓN: RAW → PROCESADO

1️⃣  DATOS RAW (p*.txt):
   - Señal RRI: Intervalos R-R del corazón (ms)
   - Formato: [timestamp, RRI_value]
   - ~30,000-40,000 samples por sujeto
   - Duración: ~2-3 horas por sujeto

2️⃣  EXTRACCIÓN DE VENTANAS:
   - Se dividen las señales en ventanas temporales
   - Probablemente ventanas de 5 minutos con overlap
   - Cada ventana → 1 fila en train/test.csv

3️⃣  EXTRACCIÓN DE FEATURES HRV:

   a) Dominio Temporal (13 features):
      - MEAN_RR, MEDIAN_RR: Valores centrales
      - SDRR, RMSSD, SDSD: Variabilidad
      - HR: Frecuencia cardíaca
      - pNN25, pNN50: Porcentaje de diferencias
      - SD1, SD2: Análisis de Poincaré
      - KURT, SKEW: Forma de distribución

   b) Dominio Frecuencia (11 features):
      - VLF, LF, HF: Potencia en bandas
      - LF/HF ratio: Balance autonómico
      - Normalized units (NU)

   c) No Lineales (2 features):
      - sampen: Sample Entropy (complejidad)
      - higuci: Dimensión fractal

4️⃣  RESULTADO:
   - ~400K ventanas procesadas
   - Solo del sujeto p2 (¿por qué?)
   - 34 features HRV por ventana
   - Labels: 'no stress', 'interruption', 'time pressure'
"""
    )


def identify_problems():
    """Identifica problemas en el dataset procesado"""
    print("\n" + "=" * 80)
    print("⚠️  PROBLEMAS IDENTIFICADOS")
    print("=" * 80)

    print(
        """
❌ PROBLEMA 1: Solo 1 de 23 sujetos procesado
   - RAW tiene 23 sujetos (p1-p25)
   - Train/Test solo tienen sujeto p2 (datasetId=2)
   - Esto limita severamente la generalización

❌ PROBLEMA 2: Train/Test del mismo sujeto
   - Ambos conjuntos son del sujeto 2
   - Alto riesgo de data leakage
   - No hay independencia entre train/test
   - Métricas sobreestimadas

❌ PROBLEMA 3: División temporal desconocida
   - No sabemos cómo se dividió train (90%) vs test (10%)
   - ¿División aleatoria? ¿Temporal? ¿Por sesión?
   - Sin información de timestamps

✅ SOLUCIÓN RECOMENDADA:

   1. Procesar TODOS los 23 sujetos desde raw
   2. Usar Leave-One-Subject-Out (LOSO):
      - Train: 22 sujetos
      - Test: 1 sujeto (rotativo)
   3. Validación cruzada de 23 folds
   4. Métricas más realistas de generalización
"""
    )


def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("🔬 ANÁLISIS COMPLETO: SWELL Dataset Pipeline")
    print("=" * 80)

    # Análisis raw
    analyze_raw_data()

    # Análisis procesado
    df_train, df_test = analyze_processed_data()

    # Explicación pipeline
    explain_pipeline()

    # Problemas
    identify_problems()

    print("\n" + "=" * 80)
    print("✅ Análisis completado")
    print("=" * 80)


if __name__ == "__main__":
    main()
