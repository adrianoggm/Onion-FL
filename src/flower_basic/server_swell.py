from __future__ import annotations

"""
SWELL central server with MQTT publishing.
Uses SwellMLP to maintain parameter names consistent with clients.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import flwr as fl
import paho.mqtt.client as mqtt

# Support running as a script (no package context)
try:
    from .swell_model import SwellMLP
except Exception:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flower_basic.swell_model import SwellMLP

MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")


class MQTTFedAvgSwell(fl.server.strategy.FedAvg):
    def __init__(self, model: SwellMLP, mqtt_client: Optional[mqtt.Client], **kwargs):
        super().__init__(**kwargs)
        self.global_model = model
        self.mqtt = mqtt_client
        self.param_names = list(model.state_dict().keys())

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, fl.common.FitRes]],
        failures,
    ) -> Optional[fl.common.Parameters]:
        print(f"\n[SERVER_SWELL] === Round {server_round} ===")
        new_parameters = super().aggregate_fit(server_round, results, failures)
        if new_parameters is None:
            print("[SERVER_SWELL] Aggregation returned None")
            return None

        try:
            # Handle different return types across Flower versions
            parameters_obj = new_parameters
            if isinstance(new_parameters, tuple):
                # Older versions may return (parameters, metrics)
                parameters_obj = new_parameters[0]

            if hasattr(parameters_obj, "tensors"):
                param_arrays = [fl.common.bytes_to_ndarray(t) for t in parameters_obj.tensors]
            else:
                param_arrays = fl.common.parameters_to_ndarrays(parameters_obj)

            state_dict = {}
            for i, name in enumerate(self.param_names):
                if i < len(param_arrays):
                    state_dict[name] = param_arrays[i]

            payload = {
                "round": server_round,
                "global_weights": {k: v.tolist() for k, v in state_dict.items()},
            }
            if self.mqtt is not None:
                self.mqtt.publish(MODEL_TOPIC, json.dumps(payload))
                print(f"[SERVER_SWELL] Published global model on {MODEL_TOPIC}")
        except Exception as e:
            print(f"[SERVER_SWELL] MQTT publish failed: {e}")
        return new_parameters


def main():
    global MQTT_BROKER, MODEL_TOPIC
    ap = argparse.ArgumentParser(description="SWELL central Flower server")
    ap.add_argument("--input_dim", type=int, required=True, help="Feature dimension (from manifest)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--server_addr", default="0.0.0.0:8080")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    ap.add_argument("--topic-global", default=MODEL_TOPIC)
    args = ap.parse_args()

    model = SwellMLP(input_dim=args.input_dim)

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        MQTT_BROKER = args.mqtt_broker
        MODEL_TOPIC = args.topic_global
        mqtt_client.connect(MQTT_BROKER, args.mqtt_port)
        mqtt_client.loop_start()
        print(f"[SERVER_SWELL] Connected to MQTT at {MQTT_BROKER}")
    except Exception as e:
        print(f"[SERVER_SWELL] MQTT connection failed: {e}")
        mqtt_client = None

    strategy = MQTTFedAvgSwell(
        model=model,
        mqtt_client=mqtt_client,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_evaluate_clients=0,
        min_available_clients=1,
    )

    print("[SERVER_SWELL] Waiting for fog clients...")
    fl.server.start_server(
        server_address=args.server_addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
