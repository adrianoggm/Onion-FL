import json
from unittest.mock import MagicMock, patch

from flower_basic.clients.baseclient import BaseMQTTComponent


def test_baseclient_subscribes_and_publishes():
    mock_client = MagicMock()

    with patch("flower_basic.clients.baseclient.mqtt.Client", return_value=mock_client):
        comp = BaseMQTTComponent(
            tag="[TEST]",
            mqtt_broker="localhost",
            mqtt_port=1883,
            subscriptions=["topic/a"],
        )

    # on_connect should subscribe to provided topics
    comp._on_connect_wrapper(mock_client, None, None, rc=0, properties=None)
    mock_client.subscribe.assert_called_with("topic/a")

    # publish_json sends a JSON payload
    comp.publish_json("topic/out", {"foo": "bar"})
    mock_client.publish.assert_called_with("topic/out", json.dumps({"foo": "bar"}))

    # stop_mqtt should stop and disconnect safely
    comp.stop_mqtt()
    mock_client.loop_stop.assert_called()
    mock_client.disconnect.assert_called()
