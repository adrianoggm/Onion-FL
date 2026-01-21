# REGLAS ESTRICTAS PARA DATASETS DE IA
## PROHIBICIONES Y POLÍTICAS OBLIGATORIAS

### ⛔ PROHIBICIONES ABSOLUTAS

#### 1. **PROHIBIDO MOCK DATA**
- **NUNCA** generar datos sintéticos/mock para algoritmos de IA
- **NUNCA** usar `np.random`, `fake`, `mock` o datos artificiales para ML/AI
- **SOLO** datos reales de SWELL y WESAD permitidos
- Violación de esta regla = ERROR CRÍTICO

#### 2. **DATASETS AUTORIZADOS**
- **ÚNICAMENTE** SWELL y WESAD
- NO otros datasets sin autorización explícita
- Usar datasets completos y originales

### ✅ POLÍTICAS OBLIGATORIAS

#### 1. **PARA EVALUACIONES** (`evaluate_*.py`)
- **OBLIGATORIO**: Usar datasets COMPLETOS (100% de los datos)
- **OBLIGATORIO**: SWELL completo desde `data/SWELL/`
- **OBLIGATORIO**: WESAD completo desde `data/WESAD/`
- **NO** limitar muestras en evaluaciones de rendimiento
- Objetivo: Obtener métricas reales y precisas

#### 2. **PARA TESTS** (`test_*.py`)
- **PERMITIDO**: Usar muestras pequeñas reales (samples/)
- **OBLIGATORIO**: Las muestras deben ser extractos auténticos
- **PROHIBIDO**: Generar datos de prueba artificiales
- Objetivo: Tests rápidos con datos reales

#### 3. **ESTRUCTURA DE DATOS**
```
data/
├── SWELL/           # Dataset completo para evaluaciones
├── WESAD/           # Dataset completo para evaluaciones  
└── samples/         # Muestras reales para tests
    ├── swell_real_sample.pkl
    └── wesad_real_sample.pkl
```

#### 4. **IMPLEMENTACIÓN**
- **Evaluadores**: Cargar datasets completos obligatoriamente
- **Tests**: Usar fallback a samples/ si dataset completo no disponible
- **Desarrollo**: NUNCA mock data, usar samples reales

### 🎯 CASOS DE USO

| Tipo de Script | Datos a Usar | Tamaño | Propósito |
|---------------|--------------|---------|-----------|
| `evaluate_*.py` | Dataset completo | 100% | Métricas finales |
| `test_*.py` | Muestras reales | ~100-1000 samples | Tests rápidos |
| `debug_*.py` | Muestras reales | Pequeñas | Debugging |
| Federated Learning | Dataset completo | 100% | Entrenamiento real |

### ⚡ ENFORCEMENT

#### Detección de Violaciones:
```python
# PROHIBIDO - Detectar y fallar
FORBIDDEN_PATTERNS = [
    'np.random.randn',
    'np.random.normal', 
    'mock_data',
    'fake_data',
    'synthetic_data',
    'generate_mock',
    'create_fake'
]
```

#### Validación Obligatoria:
- Verificar que datos provienen de archivos reales
- Confirmar rutas a datasets originales
- Rechazar cualquier generación artificial

### 📝 CUMPLIMIENTO
- **Desarrollador**: Responsable de seguir estas reglas
- **Code Review**: Verificar cumplimiento estricto  
- **CI/CD**: Fallar build si se detecta mock data
- **Documentación**: Mantener trazabilidad de datos reales

---
**IMPORTANTE**: Estas reglas garantizan la integridad científica y la validez de los resultados de IA.
