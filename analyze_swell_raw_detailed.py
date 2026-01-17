#!/usr/bin/env python3
"""
Análisis Detallado de Datos RAW del Dataset SWELL
==================================================
Analiza los archivos RRI raw y las etiquetas para entender la estructura completa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_labels():
    """Carga las etiquetas desde el archivo Excel"""
    print("\n" + "=" * 80)
    print("📋 CARGANDO ETIQUETAS (Labels)")
    print("=" * 80)
    
    labels_file = "data/SWELL/data/raw/labels/hrv stress labels.xlsx"
    
    try:
        # Leer el archivo Excel
        df_labels = pd.read_excel(labels_file)
        
        print(f"\n✓ Archivo cargado: {labels_file}")
        print(f"  Shape: {df_labels.shape}")
        print(f"\n📊 Columnas encontradas:")
        for i, col in enumerate(df_labels.columns, 1):
            print(f"  {i}. {col}")
        
        print(f"\n🔍 Primeras filas:")
        print(df_labels.head(10).to_string())
        
        print(f"\n📈 Estadísticas:")
        if 'participant' in df_labels.columns or 'subject' in df_labels.columns or 'id' in df_labels.columns:
            subj_col = [c for c in df_labels.columns if c.lower() in ['participant', 'subject', 'id', 'dataset']][0]
            print(f"  Participantes únicos: {df_labels[subj_col].nunique()}")
            print(f"  Participantes: {sorted(df_labels[subj_col].unique())}")
        
        # Buscar columnas de condición
        condition_cols = [c for c in df_labels.columns if 'condition' in c.lower() or 'stress' in c.lower() or 'label' in c.lower()]
        if condition_cols:
            print(f"\n  Columnas de condición: {condition_cols}")
            for col in condition_cols[:3]:  # Mostrar primeras 3
                if df_labels[col].dtype == 'object' or df_labels[col].nunique() < 10:
                    print(f"\n  Valores únicos en '{col}':")
                    print(f"  {df_labels[col].value_counts().to_dict()}")
        
        return df_labels
        
    except Exception as e:
        print(f"\n❌ Error al cargar etiquetas: {e}")
        return None


def analyze_rri_files():
    """Analiza todos los archivos RRI en detalle"""
    print("\n" + "=" * 80)
    print("🔬 ANÁLISIS DETALLADO DE ARCHIVOS RRI")
    print("=" * 80)
    
    raw_dir = Path("data/SWELL/data/raw/rri")
    rri_files = sorted(raw_dir.glob("p*.txt"))
    
    results = []
    
    for rri_file in rri_files:
        print(f"\n📄 Analizando {rri_file.name}...")
        
        try:
            # Cargar datos
            data = np.loadtxt(rri_file)
            subject_id = rri_file.stem
            
            # Columnas: [timestamp, RRI]
            timestamps = data[:, 0]
            rri_values = data[:, 1]
            
            # Calcular estadísticas
            duration_sec = timestamps[-1] - timestamps[0]
            duration_min = duration_sec / 60
            
            # Calcular diferencias temporales entre muestras
            time_diffs = np.diff(timestamps)
            
            # Detección de valores anómalos
            rri_mean = rri_values.mean()
            rri_std = rri_values.std()
            outliers = np.sum((rri_values < rri_mean - 3*rri_std) | (rri_values > rri_mean + 3*rri_std))
            
            # Heart rate estimado
            hr_estimated = 60000 / rri_values  # ms -> beats per minute
            
            stats = {
                'subject': subject_id,
                'num_samples': len(rri_values),
                'duration_min': duration_min,
                'duration_hours': duration_min / 60,
                'sampling_rate_hz': 1 / time_diffs.mean(),
                'rri_mean': rri_mean,
                'rri_std': rri_std,
                'rri_min': rri_values.min(),
                'rri_max': rri_values.max(),
                'rri_median': np.median(rri_values),
                'hr_mean': hr_estimated.mean(),
                'hr_std': hr_estimated.std(),
                'hr_min': hr_estimated.min(),
                'hr_max': hr_estimated.max(),
                'outliers': outliers,
                'outlier_pct': (outliers / len(rri_values)) * 100,
                'missing_beats': np.sum(time_diffs > 2.0)  # gaps > 2 segundos
            }
            
            results.append(stats)
            
            print(f"   ✓ {len(rri_values):,} muestras")
            print(f"   ✓ Duración: {duration_min:.1f} min ({duration_min/60:.2f} horas)")
            print(f"   ✓ RRI: {rri_mean:.1f} ± {rri_std:.1f} ms")
            print(f"   ✓ HR: {hr_estimated.mean():.1f} ± {hr_estimated.std():.1f} bpm")
            print(f"   ✓ Outliers: {outliers} ({stats['outlier_pct']:.2f}%)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return pd.DataFrame(results)


def create_visualizations(df_stats):
    """Crea visualizaciones de los datos raw"""
    print("\n" + "=" * 80)
    print("📊 CREANDO VISUALIZACIONES")
    print("=" * 80)
    
    output_dir = Path("swell_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Figura 1: Estadísticas generales
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SWELL Dataset RAW - Análisis por Sujeto', fontsize=16, fontweight='bold')
    
    # 1. Duración por sujeto
    ax = axes[0, 0]
    df_stats.plot(x='subject', y='duration_min', kind='bar', ax=ax, color='#3498db', legend=False)
    ax.set_title('Duración de Grabación', fontweight='bold')
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('Minutos')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 2. Número de muestras
    ax = axes[0, 1]
    df_stats.plot(x='subject', y='num_samples', kind='bar', ax=ax, color='#2ecc71', legend=False)
    ax.set_title('Número de Muestras RRI', fontweight='bold')
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('Muestras')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3. RRI medio
    ax = axes[0, 2]
    df_stats.plot(x='subject', y='rri_mean', kind='bar', ax=ax, color='#e74c3c', legend=False)
    ax.set_title('RRI Medio por Sujeto', fontweight='bold')
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('RRI (ms)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 4. Heart Rate medio
    ax = axes[1, 0]
    df_stats.plot(x='subject', y='hr_mean', kind='bar', ax=ax, color='#9b59b6', legend=False)
    ax.set_title('Frecuencia Cardíaca Media', fontweight='bold')
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('HR (bpm)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 5. Variabilidad (STD)
    ax = axes[1, 1]
    df_stats.plot(x='subject', y='rri_std', kind='bar', ax=ax, color='#f39c12', legend=False)
    ax.set_title('Variabilidad RRI (STD)', fontweight='bold')
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('STD (ms)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 6. Outliers
    ax = axes[1, 2]
    df_stats.plot(x='subject', y='outlier_pct', kind='bar', ax=ax, color='#e67e22', legend=False)
    ax.set_title('Porcentaje de Outliers', fontweight='bold')
    ax.set_xlabel('Sujeto')
    ax.set_ylabel('Outliers (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'swell_raw_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Guardado: {output_path}")
    plt.close()
    
    # Figura 2: Distribuciones
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('SWELL Dataset RAW - Distribuciones', fontsize=16, fontweight='bold')
    
    # 1. Distribución RRI
    ax = axes[0]
    ax.hist(df_stats['rri_mean'], bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_title('Distribución RRI Medio', fontweight='bold')
    ax.set_xlabel('RRI Medio (ms)')
    ax.set_ylabel('Frecuencia')
    ax.axvline(df_stats['rri_mean'].mean(), color='red', linestyle='--', 
               label=f'Media: {df_stats["rri_mean"].mean():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribución HR
    ax = axes[1]
    ax.hist(df_stats['hr_mean'], bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_title('Distribución HR Medio', fontweight='bold')
    ax.set_xlabel('HR Medio (bpm)')
    ax.set_ylabel('Frecuencia')
    ax.axvline(df_stats['hr_mean'].mean(), color='red', linestyle='--',
               label=f'Media: {df_stats["hr_mean"].mean():.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distribución duración
    ax = axes[2]
    ax.hist(df_stats['duration_hours'], bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_title('Distribución Duración Grabación', fontweight='bold')
    ax.set_xlabel('Duración (horas)')
    ax.set_ylabel('Frecuencia')
    ax.axvline(df_stats['duration_hours'].mean(), color='red', linestyle='--',
               label=f'Media: {df_stats["duration_hours"].mean():.2f}h')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'swell_raw_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_path}")
    plt.close()


def plot_example_signals():
    """Grafica señales de ejemplo de algunos sujetos"""
    print("\n" + "=" * 80)
    print("📈 GRAFICANDO SEÑALES DE EJEMPLO")
    print("=" * 80)
    
    output_dir = Path("swell_plots")
    
    # Seleccionar 6 sujetos
    subjects = ['p1', 'p2', 'p3', 'p10', 'p15', 'p20']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('SWELL Dataset RAW - Señales RRI de Ejemplo', fontsize=16, fontweight='bold')
    
    for idx, subject in enumerate(subjects):
        ax = axes[idx // 2, idx % 2]
        
        try:
            # Cargar datos
            data = np.loadtxt(f'data/SWELL/data/raw/rri/{subject}.txt')
            timestamps = data[:, 0]
            rri_values = data[:, 1]
            
            # Graficar solo primeros 10 minutos para claridad
            max_time = 600  # 10 minutos
            mask = timestamps <= (timestamps[0] + max_time)
            
            ax.plot(timestamps[mask] / 60, rri_values[mask], linewidth=0.8, alpha=0.7)
            ax.set_title(f'Sujeto {subject} (primeros 10 min)', fontweight='bold')
            ax.set_xlabel('Tiempo (minutos)')
            ax.set_ylabel('RRI (ms)')
            ax.grid(True, alpha=0.3)
            
            # Estadísticas en el gráfico
            mean_rri = rri_values[mask].mean()
            std_rri = rri_values[mask].std()
            ax.axhline(mean_rri, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.02, 0.98, f'μ={mean_rri:.1f}ms\nσ={std_rri:.1f}ms', 
                   transform=ax.transAxes, va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title(f'Sujeto {subject}', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'swell_raw_signals.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Guardado: {output_path}")
    plt.close()


def print_summary(df_stats):
    """Imprime resumen estadístico"""
    print("\n" + "=" * 80)
    print("📊 RESUMEN ESTADÍSTICO COMPLETO")
    print("=" * 80)
    
    print(f"\n🔢 Estadísticas Generales:")
    print(f"  Total sujetos: {len(df_stats)}")
    print(f"  Total muestras RRI: {df_stats['num_samples'].sum():,}")
    print(f"  Duración total: {df_stats['duration_hours'].sum():.1f} horas ({df_stats['duration_hours'].sum()/24:.1f} días)")
    
    print(f"\n⏱️  Duración por Sujeto:")
    print(f"  Media: {df_stats['duration_min'].mean():.1f} min ({df_stats['duration_hours'].mean():.2f} h)")
    print(f"  Mínima: {df_stats['duration_min'].min():.1f} min")
    print(f"  Máxima: {df_stats['duration_min'].max():.1f} min")
    print(f"  STD: {df_stats['duration_min'].std():.1f} min")
    
    print(f"\n💓 RRI (Intervalos R-R):")
    print(f"  Media global: {df_stats['rri_mean'].mean():.1f} ms")
    print(f"  Rango: {df_stats['rri_min'].min():.1f} - {df_stats['rri_max'].max():.1f} ms")
    print(f"  Variabilidad media (STD): {df_stats['rri_std'].mean():.1f} ms")
    
    print(f"\n❤️  Frecuencia Cardíaca (HR):")
    print(f"  Media global: {df_stats['hr_mean'].mean():.1f} bpm")
    print(f"  Rango: {df_stats['hr_min'].min():.1f} - {df_stats['hr_max'].max():.1f} bpm")
    print(f"  STD media: {df_stats['hr_std'].mean():.1f} bpm")
    
    print(f"\n⚠️  Calidad de Datos:")
    print(f"  Outliers promedio: {df_stats['outlier_pct'].mean():.2f}%")
    print(f"  Missing beats promedio: {df_stats['missing_beats'].mean():.1f}")
    print(f"  Sujetos con >1% outliers: {(df_stats['outlier_pct'] > 1).sum()}")
    
    print(f"\n📋 Tabla Completa de Estadísticas:")
    print(df_stats[['subject', 'num_samples', 'duration_min', 'rri_mean', 'hr_mean', 
                    'outlier_pct']].to_string(index=False))


def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("🔬 ANÁLISIS DETALLADO DE DATOS RAW - SWELL DATASET")
    print("=" * 80)
    
    # 1. Cargar labels
    df_labels = load_labels()
    
    # 2. Analizar archivos RRI
    df_stats = analyze_rri_files()
    
    # 3. Imprimir resumen
    print_summary(df_stats)
    
    # 4. Crear visualizaciones
    create_visualizations(df_stats)
    
    # 5. Graficar señales de ejemplo
    plot_example_signals()
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)
    print(f"\n📁 Visualizaciones guardadas en: swell_plots/")
    print("   - swell_raw_analysis.png")
    print("   - swell_raw_distributions.png")
    print("   - swell_raw_signals.png")


if __name__ == "__main__":
    main()
