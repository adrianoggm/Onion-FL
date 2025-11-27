import importlib


def test_entrypoints_expose_main():
    modules = [
        "flower_basic.clients.swell",
        "flower_basic.clients.fog_bridge_swell",
        "flower_basic.servers.swell",
        "flower_basic.brokers.fog",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main"), f"{mod_name} should expose main()"
