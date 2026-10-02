import importlib


def test_entrypoints_expose_main():
    modules = [
        "onion_fl.clients.swell",
        "onion_fl.clients.fog_bridge_swell",
        "onion_fl.servers.swell",
        "onion_fl.brokers.fog",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main"), f"{mod_name} should expose main()"
