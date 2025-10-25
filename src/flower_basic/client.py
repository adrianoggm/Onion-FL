"""
client.py - Cliente Local de Aprendizaje Federado

Este mÃ³dulo implementa clientes locales que:
1. Entrenan modelos CNN en datos WESAD locales
2. Publican actualizaciones de modelo vÃ­a MQTT al broker fog
3. Reciben modelos globales del servidor central vÃ­a MQTT

ARQUITECTURA:
- Cada cliente entrena en su particiÃ³n local de datos WESAD
- EnvÃ­a actualizaciones al broker fog vÃ­a topic 'fl/updates'
- El broker fog agrega K clientes por regiÃ³n antes de enviar al servidor central
- Recibe modelos globales actualizados vÃ­a topic 'fl/global_model'

FLUJO:
1. Cliente entrena modelo local por 1 Ã©poca
2. Publica pesos actualizados al broker fog vÃ­a MQTT
3. Espera nuevo modelo global del servidor central
4. Actualiza modelo local y repite
"""

import json
import os
import time

import paho.mqtt.client as mqtt
import torch

from .datasets import load_wesad_dataset
from .model import ECGModel

# -----------------------------------------------------------------------------
# CONFIGURACIÃN MQTT
# -----------------------------------------------------------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_UPDATES = os.getenv("MQTT_TOPIC_UPDATES", "fl/updates")  # Topic para publicar actualizaciones locales
TOPIC_GLOBAL_MODEL = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")  # Topic para recibir modelos globales


# -----------------------------------------------------------------------------
# CLIENTE LOCAL MQTT
# -----------------------------------------------------------------------------
class FLClientMQTT:
    """
    Cliente de aprendizaje federado que usa MQTT para comunicaciÃ³n.

    Funciones principales:
    1. Entrenamiento local en datos WESAD particionados
    2. PublicaciÃ³n de actualizaciones al broker fog
    3. RecepciÃ³n de modelos globales del servidor central
    4. SincronizaciÃ³n con la arquitectura fog computing
    """

    def __init__(self):
        # Inicializar modelo CNN para ECG
        self.model = ECGModel()

        # Cargar y preparar datos locales ECG5000
        X_train, X_test, y_train, y_test = load_wesad_dataset()
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            ),
            batch_size=32,
            shuffle=True,
        )

        # Configurar cliente MQTT para comunicaciÃ³n con broker fog
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        print(f"[CLIENT] Conectado a broker MQTT en {MQTT_BROKER}:{MQTT_PORT}")

        # Iniciar loop MQTT en hilo separado
        self.mqtt.loop_start()

        # Flag para detectar cuando llega un nuevo modelo global
        self._got_global = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback MQTT al conectarse: suscribirse a modelos globales."""
        print(
            f"[CLIENT] MQTT conectado (rc={rc}), suscribiÃ©ndose a {TOPIC_GLOBAL_MODEL}"
        )
        client.subscribe(TOPIC_GLOBAL_MODEL)

    def _on_message(self, client, userdata, msg):
        """
        Callback MQTT para procesar modelos globales del servidor central.

        Recibe el modelo global actualizado y lo carga en el modelo local
        para la prÃ³xima ronda de entrenamiento.
        """
        if msg.topic == TOPIC_GLOBAL_MODEL:
            try:
                payload = json.loads(msg.payload.decode())
                # payload contiene: {"round": X, "global_weights": {param_name:
                # [values]}}
                if "global_weights" in payload:
                    weights_dict = payload["global_weights"]
                    state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
                    self.model.load_state_dict(state_dict, strict=True)
                    round_num = payload.get("round", "?")
                    print(f"[CLIENT] Modelo global cargado de ronda {round_num}")
                    self._got_global = True
                else:
                    print("[CLIENT] Formato de payload invÃ¡lido recibido")
            except Exception as e:
                print(f"[CLIENT] Error procesando mensaje MQTT: {e}")

    def train_one_round(self):
        """
        Ejecuta una ronda de entrenamiento local y publica la actualizaciÃ³n.

        Pasos:
        1. Entrena el modelo local por 1 Ã©poca en datos WESAD
        2. Serializa los pesos actualizados
        3. Publica la actualizaciÃ³n al broker fog vÃ­a MQTT
        4. Espera el prÃ³ximo modelo global del servidor central
        """
        # Entrenamiento local por 1 Ã©poca
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()  # Para clasificaciÃ³n binaria ECG
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for X, y in self.train_loader:
            optimizer.zero_grad()
            # AÃ+/-adir dimensiÃ³n de canal para CNN: (batch, 1, sequence_length)
            X = X.unsqueeze(1)
            logits = self.model(X).squeeze()  # Eliminar dimensiones extra
            loss = criterion(logits, y.float())  # Convertir a float para BCE
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[CLIENT] Entrenamiento completado. Loss promedio: {avg_loss:.4f}")

        # Serializar pesos actuales y publicar con metadata
        state = self.model.state_dict()
        payload = {
            "client_id": f"client_{id(self) % 1000}",  # ID simple del cliente
            "region": "region_0",  # RegiÃ³n por defecto (puede parametrizarse)
            "weights": {k: v.cpu().numpy().tolist() for k, v in state.items()},
            "num_samples": (
                len(self.train_loader.dataset)
                if hasattr(self.train_loader.dataset, "__len__")
                else len(self.train_loader)
            ),
            "loss": avg_loss,
        }
        # Region override via env (MQTT_REGION) if provided
        payload["region"] = os.getenv("MQTT_REGION", payload["region"])  

        self.mqtt.publish(TOPIC_UPDATES, json.dumps(payload))
        print(f"[CLIENT] ActualizaciÃ³n local publicada en {TOPIC_UPDATES}")

        # Esperar hasta que el broker fog publique un nuevo modelo global
        print("[CLIENT] Esperando nuevo modelo global...")
        while not self._got_global:
            time.sleep(1)
        self._got_global = False
        print("[CLIENT] Nuevo modelo global recibido, listo para prÃ³xima ronda")

    def run(self, rounds: int = 5, delay: float = 5.0):
        """
        Bucle principal del cliente: entrenarâpublicarâsincronizarârepetir.

        Args:
            rounds: NÃºmero de rondas de entrenamiento federado
            delay: Delay entre rondas (segundos)
        """
        print(f"[CLIENT] Iniciando {rounds} rondas de aprendizaje federado")
        for rnd in range(1, rounds + 1):
            print(f"\n=== Ronda {rnd}/{rounds} ===")
            self.train_one_round()
            if rnd < rounds:  # No hacer delay en la Ãºltima ronda
                time.sleep(delay)

        # Limpieza de conexiones
        print("[CLIENT] Entrenamiento federado completado, cerrando conexiones")
        self.mqtt.loop_stop()
        self.mqtt.disconnect()


# -----------------------------------------------------------------------------
# FUNCIÃN PRINCIPAL: Ejecutar cliente local
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Punto de entrada principal para ejecutar un cliente local.

    Configura y ejecuta un cliente que:
    - Entrena en datos WESAD locales
    - Participa en aprendizaje federado vÃ­a MQTT
    - Se comunica con la arquitectura fog computing
    """
    print("=== CLIENTE LOCAL DE APRENDIZAJE FEDERADO ===")
    print("Iniciando cliente MQTT para fog computing...")

    client = FLClientMQTT()
    client.run(rounds=3, delay=2.0)
