# Diagrama de Clases ASCII - Flower-Basic

## 📊 Jerarquía de Herencia (Árbol)

```
┌─────────────────────────────────────────────────────────────────────┐
│                       CLASES BASE (Abstractas)                      │
└──────────────┬──────────────────────────────┬──────────────────────┘
               │                              │
       ┌───────▼────────┐            ┌────────▼─────────┐
       │ BaseMQTTCmpt   │            │ fl.client.NumPy  │
       │                │            │ Client           │
       │ (paho.mqtt)    │            │                  │
       └───────┬────────┘            └────────┬─────────┘
               │                              │
     ┌─────────┴─────────┐                    │
     │                   │                    │
 ┌───▼──────┐   ┌───────▼──────┐             │
 │ Swell    │   │ Sweet        │             │
 │ FLClient │   │ FLClient     │             │
 │ MQTT     │   │ MQTT         │             │
 └──────────┘   └──────────────┘    ┌────────▼────────┐
                                     │                 │
                              ┌──────▼────┐   ┌───────▼────┐
                              │ FogClient  │   │ FogClient  │
                              │ Swell      │   │ Sweet      │
                              │            │   │            │
                              │ (MQTT+FL)  │   │ (MQTT+FL)  │
                              └────────────┘   └────────────┘


┌──────────────────────────────────────────────────────────┐
│            Servidores (Flower + MQTT)                   │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────▼──────────┐
       │ fl.server.strategy│
       │ FedAvg (Flower)   │
       └───────┬──────────┘
               │
     ┌─────────┴──────────┐
     │                    │
 ┌───▼──────┐    ┌───────▼──────┐
 │ MQTTFedAvg    │ MQTTFedAvg   │
 │ Swell         │ Sweet        │
 │              │              │
 │ (publica en  │ (publica en  │
 │  MQTT)       │  MQTT)       │
 └──────────────┘ └──────────────┘
```

---

## 🔌 Relaciones de Composición

```
┌────────────────────────────────────────┐
│     SwellFLClientMQTT                  │
│  (Client Edge - SWELL)                 │
├────────────────────────────────────────┤
│ Atributos:                             │
│  • node_dir: Path                      │
│  • region: str (ej: "fog_0")           │
│  • model: SwellMLP                     │ ──┐
│  • train_loader: DataLoader            │   │ Composición
│  • global_model: dict                  │   │ (usa modelo)
│  • mqtt: mqtt.Client                   │ ──┤
│  • topic_updates: str                  │   │
│  • topic_global: str                   │ ──┤
├────────────────────────────────────────┤   │
│ Métodos:                               │   │
│  • train_local() → weights             │   │
│  • on_message(msg) → ∅                 │   │
│  • publish_update() → ∅                │   │
│  • process_global_weights() → ∅        │   │
│  • run_rounds() → ∅                    │   │
└────────────────────────────────────────┘   │
                                             │
                                    ┌────────▼─────────┐
                                    │   SwellMLP       │
                                    │   (pytorch)      │
                                    ├──────────────────┤
                                    │ • input_dim      │
                                    │ • hidden_dim     │
                                    │ • forward()      │
                                    │ • state_dict()   │
                                    └──────────────────┘

┌────────────────────────────────────────┐
│     SweetFLClientMQTT                  │
│  (Client Edge - SWEET)                 │
├────────────────────────────────────────┤
│ Atributos:                             │
│  • node_dir: Path                      │
│  • region: str (ej: "fog_0")           │
│  • subject_id: str (opcional)          │
│  • model: SweetMLP                     │ ──┐
│  • train_loader: DataLoader            │   │
│  • global_model: dict                  │   │
│  • mqtt: mqtt.Client                   │   │
│  • topic_updates: str                  │ ──┤
│  • topic_global: str                   │   │
├────────────────────────────────────────┤   │
│ Métodos: (similares a Swell)           │   │
└────────────────────────────────────────┘   │
                                             │
                                    ┌────────▼──────────┐
                                    │   SweetMLP       │
                                    │   (pytorch)      │
                                    ├───────────────────┤
                                    │ • input_dim       │
                                    │ • hidden_dims[]   │
                                    │ • num_classes     │
                                    │ • forward()       │
                                    │ • state_dict()    │
                                    └───────────────────┘

┌────────────────────────────────────────┐
│     FogClientSwell                     │
│  (Fog Bridge - Flower + MQTT)          │
├────────────────────────────────────────┤
│ Hereda:                                │
│  • BaseMQTTComponent (MQTT)            │
│  • fl.client.NumPyClient (Flower)      │
├────────────────────────────────────────┤
│ Atributos:                             │
│  • model: SwellMLP                     │ ──┐
│  • partial_weights: dict               │   │ Composición
│  • region: str                         │   │
│  • mqtt: mqtt.Client                   │ ──┤
│  • server_address: str                 │   │
├────────────────────────────────────────┤   │
│ Métodos:                               │   │
│  • fit(parameters, config)             │   │
│  • get_parameters(config)              │   │
│  • on_message(msg) [escucha MQTT]      │   │
│  • evaluate(parameters, config)        │   │
└────────────────────────────────────────┘   │
                                             │
                                    ┌────────▼─────────┐
                                    │   SwellMLP       │
                                    │   (mismo modelo) │
                                    └──────────────────┘

┌────────────────────────────────────────┐
│     FogClientSweet                     │
│  (Fog Bridge - Flower + MQTT)          │
├────────────────────────────────────────┤
│ Hereda:                                │
│  • BaseMQTTComponent (MQTT)            │
│  • fl.client.NumPyClient (Flower)      │
├────────────────────────────────────────┤
│ Atributos:                             │
│  • model: SweetMLP                     │ ──┐
│  • partial_weights: dict               │   │
│  • region: str                         │   │
│  • mqtt: mqtt.Client                   │ ──┤
│  • server_address: str                 │   │
├────────────────────────────────────────┤   │
│ Métodos: (similares a FogClientSwell)  │   │
└────────────────────────────────────────┘   │
                                             │
                                    ┌────────▼──────────┐
                                    │   SweetMLP       │
                                    │   (mismo modelo) │
                                    └───────────────────┘

┌────────────────────────────────────────┐
│     MQTTFedAvgSwell                    │
│  (Servidor Central - Flower + MQTT)    │
├────────────────────────────────────────┤
│ Hereda:                                │
│  • fl.server.strategy.FedAvg           │
├────────────────────────────────────────┤
│ Atributos:                             │
│  • model: SwellMLP                     │ ──┐
│  • mqtt_client: mqtt.Client            │   │ Composición
│  • eval_data: Tuple                    │   │
│  • total_rounds: int                   │ ──┤
├────────────────────────────────────────┤   │
│ Métodos:                               │   │
│  • configure_fit(rnd, parameters, ...) │   │
│  • aggregate_fit(rnd, results, ...)    │   │
│  • evaluate(rnd, parameters)           │   │
│  • publish_global_model(weights)       │   │
└────────────────────────────────────────┘   │
                                             │
                                    ┌────────▼─────────┐
                                    │   SwellMLP       │
                                    │   (mismo modelo) │
                                    └──────────────────┘

┌────────────────────────────────────────┐
│     MQTTFedAvgSweet                    │
│  (Servidor Central - Flower + MQTT)    │
├────────────────────────────────────────┤
│ Hereda:                                │
│  • fl.server.strategy.FedAvg           │
├────────────────────────────────────────┤
│ Atributos:                             │
│  • model: SweetMLP                     │ ──┐
│  • mqtt_client: mqtt.Client            │   │ Composición
│  • eval_data: Tuple                    │   │
│  • total_rounds: int                   │ ──┤
├────────────────────────────────────────┤   │
│ Métodos: (similares a MQTTFedAvgSwell) │   │
└────────────────────────────────────────┘   │
                                             │
                                    ┌────────▼──────────┐
                                    │   SweetMLP       │
                                    │   (mismo modelo) │
                                    └───────────────────┘
```

---

## 📡 Flujo de Comunicación MQTT

```
┌─────────────────────────────────────────────────────────────────┐
│                      MQTT Topics                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOPIC: fl/updates                                              │
│  Publicadores:                                                  │
│    - SwellFLClientMQTT (clientes edge)                          │
│    - SweetFLClientMQTT (clientes edge)                          │
│                                                                 │
│  Suscriptores:                                                  │
│    - fog.py (broker regional)                                   │
│    - sweet_fog.py (broker regional SWEET)                       │
│                                                                 │
│  Payload:                                                       │
│  {                                                              │
│    "client_id": "swell_1",                                      │
│    "region": "fog_0",                                           │
│    "num_samples": 150,                                          │
│    "weights": {model_params},                                   │
│    "trace_context": {...}                                       │
│  }                                                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOPIC: fl/partial                                              │
│  Publicadores:                                                  │
│    - fog.py (broker regional)                                   │
│    - sweet_fog.py (broker regional SWEET)                       │
│                                                                 │
│  Suscriptores:                                                  │
│    - FogClientSwell (fog bridge)                                │
│    - FogClientSweet (fog bridge)                                │
│                                                                 │
│  Payload:                                                       │
│  {                                                              │
│    "region": "fog_0",                                           │
│    "partial_weights": {aggregated_params},  ← Promedio NumPy   │
│    "total_samples": 450,                                        │
│    "timestamp": 1704547200.123,                                 │
│    "trace_context": {...}                                       │
│  }                                                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOPIC: fl/global_model                                         │
│  Publicadores:                                                  │
│    - MQTTFedAvgSwell (servidor central)                         │
│    - MQTTFedAvgSweet (servidor central)                         │
│                                                                 │
│  Suscriptores:                                                  │
│    - SwellFLClientMQTT (clientes edge)                          │
│    - SweetFLClientMQTT (clientes edge)                          │
│                                                                 │
│  Payload:                                                       │
│  {                                                              │
│    "round": 1,                                                  │
│    "weights": {global_model_params},                            │
│    "accuracy": 0.92,                                            │
│    "loss": 0.15,                                                │
│    "timestamp": 1704547300.456,                                 │
│    "trace_context": {...}                                       │
│  }                                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Flujo de Datos (Ronda FL)

```
RONDA 1:
═════════════════════════════════════════════════════════════════

1. ENTRENAMIENTO LOCAL (Edge Layer)
┌────────────────────┐
│ SwellFLClientMQTT  │  train_local()
│ region="fog_0"     │  ├─ Load train_loader
│                    │  ├─ Inicializa con global_weights (ronda 1: init)
│                    │  ├─ Forward pass
│                    │  ├─ Compute loss
│                    │  ├─ Backward pass
│                    │  └─ Return weights
└────────┬───────────┘
         │
         │ publish_update()
         │ Topic: fl/updates
         │ Payload: {client_id, region, weights, num_samples}
         │
         ▼
2. BUFFERIZACIÓN (Fog Broker Layer)
┌──────────────────────┐
│ fog.py Broker        │
│ region="fog_0"       │  on_update()
│                      │  ├─ buffers["fog_0"].append(weights)
│                      │  ├─ Check: len(buffers) >= K (default K=3)?
│                      │  └─ if NO → wait
└──────────┬───────────┘
           │
           │ (Cuando K=3 clientes han publicado...)
           │
           ▼
3. AGREGACIÓN (Fog Broker - NumPy)
┌──────────────────────┐
│ fog.py Broker        │
│ region="fog_0"       │  weighted_average()
│                      │  ├─ w1, w2, w3 = buffers["fog_0"]
│                      │  ├─ weights = [n1/N, n2/N, n3/N]
│                      │  ├─ partial = Σ(wi * θi)  ← FedAvg NumPy
│                      │  └─ Clear buffer
└──────────┬───────────┘
           │
           │ publish(fl/partial)
           │ Payload: {region, partial_weights, total_samples}
           │
           ▼
4. FORWARDING A FLOWER (Fog Bridge Layer)
┌────────────────────────┐
│ FogClientSwell         │
│ region="fog_0"         │  on_message() escucha fl/partial
│                        │  ├─ self.partial_weights = data
│                        │  └─ wait para próximo Flower setup
└────────────┬───────────┘
             │
             │ Flower llama fit(parameters, config)
             │
             ▼
5. AGREGACIÓN GLOBAL (Central Server Layer)
┌──────────────────────┐
│ MQTTFedAvgSwell      │
│ Servidor Flower      │  aggregate_fit()
│                      │  ├─ Recibe [partial_fog_0, partial_fog_1]
│                      │  ├─ global = FedAvg(partials)  ← Flower
│                      │  └─ Evalúa en test data
└──────────┬───────────┘
           │
           │ publish_global_model()
           │ Topic: fl/global_model
           │ Payload: {round, weights, accuracy, loss}
           │
           ▼
6. BROADCAST (Back to Edge)
┌────────────────────┐
│ SwellFLClientMQTT  │  on_message() escucha fl/global_model
│ region="fog_0"     │  ├─ self.global_model = data
│                    │  └─ Ready para siguiente ronda
└────────────────────┘

RONDA 2: (Repetir desde paso 1, pero inicializando con global_model)
═════════════════════════════════════════════════════════════════
```

---

## 📊 Tabla de Tipos de Datos

| Clase | Atributo | Tipo | Descripción |
|-------|----------|------|-------------|
| `SwellFLClientMQTT` | `model` | `SwellMLP` | Modelo local |
| | `region` | `str` | Identificador de región (ej: "fog_0") |
| | `train_loader` | `DataLoader` | Datos de entrenamiento |
| | `global_model` | `dict` | Parámetros recibidos del servidor |
| | `mqtt` | `mqtt.Client` | Cliente MQTT (vía herencia) |
| `FogClientSwell` | `partial_weights` | `dict` | Pesos agregados del broker |
| | `model` | `SwellMLP` | Mismo modelo que clientes |
| | `server_address` | `str` | Dirección del servidor Flower |
| `MQTTFedAvgSwell` | `model` | `SwellMLP` | Modelo global |
| | `eval_data` | `Tuple[np.ndarray, np.ndarray]` | Datos de evaluación |
| | `total_rounds` | `int` | Número de rondas FL |

---

## 🎯 Patrones de Diseño

### 1. **Template Method Pattern**
```
BaseMQTTComponent
├─ on_message() [hook abstracto]
│
└─ Subclases implementan on_message():
   ├─ SwellFLClientMQTT → procesa global_model
   ├─ SweetFLClientMQTT → procesa global_model
   ├─ FogClientSwell → procesa partial_weights
   └─ FogClientSweet → procesa partial_weights
```

### 2. **Strategy Pattern**
```
fl.server.strategy.FedAvg [Flower base strategy]
│
├─ MQTTFedAvgSwell [extiende con MQTT publishing]
└─ MQTTFedAvgSweet [extiende con MQTT publishing]
```

### 3. **Adapter Pattern**
```
FogClientSwell / FogClientSweet
├─ Adaptan: pesos MQTT → formato Flower NumPyClient
├─ Escuchan: fl/partial (MQTT)
└─ Entregan: fit() result (Flower gRPC)
```

### 4. **Builder Pattern**
```
Configuración YAML → FederatedArchitecture → Comandos Docker
```

---

## 🔐 Interfaces Clave

### `fl.client.NumPyClient` (Flower)
```python
class NumPyClient:
    def fit(self, parameters: List[np.ndarray], config) 
        → Tuple[List[np.ndarray], int, dict]
    
    def evaluate(self, parameters: List[np.ndarray], config)
        → Tuple[float, int, dict]
    
    def get_parameters(self, config) 
        → List[np.ndarray]
```

### `fl.server.strategy.FedAvg` (Flower)
```python
class FedAvg:
    def configure_fit(self, rnd: int, parameters, ...)
        → Tuple[List, dict]
    
    def aggregate_fit(self, rnd: int, results, failures)
        → Tuple[parameters, metrics]
    
    def evaluate(self, rnd: int, parameters)
        → Tuple[loss, metrics]
```

### `paho.mqtt.client.Client` (MQTT)
```python
class Client:
    def connect(self, host, port, keepalive)
        → None
    
    def publish(self, topic: str, payload)
        → None
    
    def subscribe(self, topic: str)
        → None
    
    def on_message(self, func)
        → None  [callback]
    
    def loop_start()
        → None  [background thread]
```

---

## 🚀 Instanciación en Docker

```
Docker Compose Services:
├─ mosquitto
│  └─ image: eclipse-mosquitto:2
│     port: 1883
│
├─ server (MQTTFedAvgSwell)
│  ├─ depends_on: mosquitto
│  ├─ args: [--input_dim, --rounds, --mqtt-broker mosquitto, ...]
│  └─ address: 0.0.0.0:8080
│
├─ fog_bridge_fog_0 (FogClientSwell)
│  ├─ depends_on: mosquitto
│  ├─ args: [--region fog_0, --mqtt-broker mosquitto, ...]
│  └─ suscribe: fl/partial
│
├─ fog_bridge_fog_1 (FogClientSwell)
│  ├─ depends_on: mosquitto
│  ├─ args: [--region fog_1, --mqtt-broker mosquitto, ...]
│  └─ suscribe: fl/partial
│
├─ broker (fog.py)
│  ├─ depends_on: mosquitto
│  ├─ env: [MQTT_BROKER mosquitto, FOG_K 2, ...]
│  └─ suscribe: fl/updates → publica: fl/partial
│
├─ client_fog0_c1 (SwellFLClientMQTT)
│  ├─ depends_on: mosquitto, broker
│  ├─ args: [--region fog_0, --mqtt-broker mosquitto, ...]
│  └─ publica: fl/updates, suscribe: fl/global_model
│
├─ client_fog0_c2 (SwellFLClientMQTT)
│  ├─ depends_on: mosquitto, broker
│  ├─ args: [--region fog_0, --mqtt-broker mosquitto, ...]
│  └─ publica: fl/updates, suscribe: fl/global_model
│
└─ client_fog1_c1 (SwellFLClientMQTT)
   ├─ depends_on: mosquitto, broker
   ├─ args: [--region fog_1, --mqtt-broker mosquitto, ...]
   └─ publica: fl/updates, suscribe: fl/global_model
```
