"""Tests for the generic ``torch_module`` family and its presets (issue #245).

Keys, input rebuilding, aggregation and determinism use the weights the models
are initialised with, plus zero batches as shape fixtures; nothing is trained on
invented data. The tests that train use the real FEMNIST extract and are
skipped when it is absent (docs/RULES.md).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from onion_fl.core.registry import PluginError
from onion_fl.learning.aggregators import Contribution, aggregators
from onion_fl.learning.model import (
    DataShape,
    load_arrays,
    models,
    param_groups,
    state_arrays,
)
from onion_fl.learning.torch_module import (
    PRESETS,
    TRUSTED_ENV,
    TorchModule,
    TorchModuleError,
)

FEMNIST = DataShape(dataset="femnist", task="character", n_features=784, n_classes=62)
CIFAR = DataShape(dataset="cifar10", task="object", n_features=3072, n_classes=10)
SHAKESPEARE = DataShape(
    dataset="shakespeare", task="next_char", n_features=80, n_classes=80
)
AUDIO = DataShape(dataset="speech", task="command", n_features=1600, n_classes=12)

# preset -> (its config, a dataset it fits, a second dataset with another task)
CASES = {
    "leaf_femnist_cnn": ({}, FEMNIST, ("emnist", "digit", 10)),
    "niid_bench_cnn": ({}, CIFAR, ("svhn", "digit", 10)),
    "leaf_shakespeare_lstm": ({}, SHAKESPEARE, ("sonnets", "next_char", 80)),
    "audio_cnn1d": ({"input_shape": [1, 1600]}, AUDIO, ("urban", "sound", 10)),
}


def family(module: str, **config):
    return models.create("torch_module", {"module": module, **config})


def build(module: str, *shapes: DataShape, seed: int = 0, **config) -> TorchModule:
    return family(module, **config).build(list(shapes), seed=seed)


def preset(name: str, *, second: bool = False, seed: int = 0) -> TorchModule:
    config, shape, (dataset, task, classes) = CASES[name]
    shapes = [shape]
    if second:
        shapes.append(shape.model_copy(update={"dataset": dataset, "task": task}))
        shapes[-1] = shapes[-1].model_copy(update={"n_classes": classes})
    return build(name, *shapes, seed=seed, **config)


def groups(model: nn.Module) -> set[str]:
    return set(param_groups(model.state_dict()))


def test_every_preset_is_covered() -> None:
    assert set(CASES) == set(PRESETS)


# --- keys follow the namespace contract ------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_preset_edge_holds_its_adapter_the_trunk_and_its_head(name: str) -> None:
    _, shape, _ = CASES[name]

    assert groups(preset(name)) == {
        f"adapter.{shape.dataset}",
        "trunk",
        f"head.{shape.task}",
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_preset_global_model_shares_the_trunk(name: str) -> None:
    _, shape, (dataset, task, _) = CASES[name]

    assert groups(preset(name, second=True)) == {
        f"adapter.{shape.dataset}",
        f"adapter.{dataset}",
        "trunk",
        f"head.{shape.task}",
        f"head.{task}",
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_preset_gives_one_logit_per_class_for_each_dataset(name: str) -> None:
    model = preset(name, second=True).eval()
    _, shape, (dataset, _, classes) = CASES[name]

    assert model(torch.zeros((3, shape.n_features)), dataset=shape.dataset).shape == (
        3,
        shape.n_classes,
    )
    assert model(torch.zeros((3, shape.n_features)), dataset=dataset).shape == (
        3,
        classes,
    )
    with pytest.raises(ValueError, match="pass dataset"):
        model(torch.zeros((3, shape.n_features)))


def test_the_presets_keep_the_papers_sizes() -> None:
    femnist = preset("leaf_femnist_cnn").state_dict()
    cifar = preset("niid_bench_cnn").state_dict()
    lstm = preset("leaf_shakespeare_lstm").state_dict()

    assert femnist["adapter.femnist.0.weight"].shape == (32, 1, 5, 5)
    assert femnist["trunk.4.weight"].shape == (2048, 7 * 7 * 64)
    assert femnist["head.character.weight"].shape == (62, 2048)
    assert cifar["adapter.cifar10.0.weight"].shape == (6, 3, 5, 5)
    assert cifar["trunk.4.weight"].shape == (120, 16 * 5 * 5)
    assert cifar["head.object.weight"].shape == (10, 84)
    assert lstm["adapter.shakespeare.0.embedding.weight"].shape == (80, 8)
    assert lstm["trunk.0.lstm.weight_hh_l1"].shape == (4 * 256, 256)
    assert lstm["head.next_char.weight"].shape == (80, 256)


def test_an_edge_loads_only_its_keys_from_the_global_state() -> None:
    edge = preset("niid_bench_cnn", seed=1)

    loaded = load_arrays(edge, state_arrays(preset("niid_bench_cnn", second=True)))

    assert sorted(loaded) == sorted(edge.state_dict())


# --- the input is rebuilt from the x_<i> columns ---------------------------------------


def test_kept_columns_go_back_to_their_pixels_and_dropped_ones_are_zero() -> None:
    kept = DataShape(
        dataset="femnist",
        task="character",
        n_features=3,
        n_classes=62,
        features=("x_0", "x_29", "x_783"),
    )
    model = build("leaf_femnist_cnn", kept)

    image = model.example(torch.tensor([[1.0, 2.0, 3.0]]))

    assert image.shape == (1, 1, 28, 28)
    assert (
        image[0, 0, 0, 0] == 1 and image[0, 0, 1, 1] == 2 and image[0, 0, 27, 27] == 3
    )
    assert image.abs().sum() == 6


def test_columns_in_their_order_are_reshaped_as_they_are() -> None:
    named = FEMNIST.model_copy(update={"features": tuple(f"x_{i}" for i in range(784))})
    model = build("leaf_femnist_cnn", named)
    x = torch.arange(784, dtype=torch.float32).reshape(1, 784)

    assert model.columns["femnist"] is None
    torch.testing.assert_close(model.example(x), x.reshape(1, 1, 28, 28))


@pytest.mark.parametrize(
    "shape, match",
    [
        (FEMNIST.model_copy(update={"n_features": 700}), "do not fill"),
        (
            FEMNIST.model_copy(
                update={"n_features": 2, "features": ("x_0", "pixel_1")}
            ),
            "x_<i>",
        ),
        (
            FEMNIST.model_copy(update={"n_features": 1, "features": ("x_784",)}),
            "outside",
        ),
        (
            FEMNIST.model_copy(update={"n_features": 2, "features": ("x_3", "x_3")}),
            "repeated",
        ),
    ],
)
def test_columns_that_cannot_rebuild_the_input_are_rejected(
    shape: DataShape, match: str
) -> None:
    with pytest.raises(TorchModuleError, match=match):
        build("leaf_femnist_cnn", shape)


def test_an_input_the_module_cannot_take_fails_at_build(trusted) -> None:
    with pytest.raises(ValueError, match="H and W divisible by 4"):
        build("leaf_femnist_cnn", FEMNIST, input_shape=[1, 27, 29])
    with pytest.raises(TorchModuleError, match="fails on femnist"):
        wide = FEMNIST.model_copy(update={"n_features": 900})
        build(f"{HERE}:Fixed", wide, input_shape=[1, 30, 30])
    with pytest.raises(TorchModuleError, match=r"expected \(batch, 62\)"):
        build(f"{HERE}:TenLogits", FEMNIST)


def test_feature_names_must_match_n_features() -> None:
    with pytest.raises(ValueError, match="feature names"):
        DataShape(dataset="d", task="t", n_features=2, n_classes=2, features=("x_0",))


# --- registry, config and external classes -------------------------------------------


def test_the_registry_describes_it() -> None:
    entry = next(e for e in models.describe() if e["name"] == "torch_module")

    assert entry["title"] == "Módulo PyTorch"
    assert set(entry["params"]["properties"]) == {"module", "params", "input_shape"}


@pytest.mark.parametrize(
    "params, match",
    [
        ({"module": "resnet152"}, "unknown module"),
        ({"module": "torch.nn:Linear"}, "not trusted"),
        ({"module": "os; rm:x"}, "unknown module"),
        ({"module": "audio_cnn1d"}, "no default input_shape"),
        ({"module": "leaf_femnist_cnn", "params": {"depth": 3}}, "invalid params"),
        ({"module": "leaf_femnist_cnn", "input_shape": [0, 28]}, "input_shape"),
        ({"module": "leaf_femnist_cnn", "trunk": "shared"}, "trunk"),
    ],
)
def test_invalid_configs_are_rejected(params: dict, match: str) -> None:
    with pytest.raises(PluginError, match=match):
        models.create("torch_module", params)


class External(nn.Module):
    """A model written outside the framework: a linear adapter, MLP trunk, head."""

    input_shape = (1, 28, 28)

    def __init__(self, shapes, input_shape, width: int = 16) -> None:
        super().__init__()
        (shape,) = shapes
        n = int(np.prod(input_shape))
        self.adapter = nn.ModuleDict(
            {shape.dataset: nn.Sequential(nn.Flatten(), nn.Linear(n, width))}
        )
        self.trunk = nn.Sequential(nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.ModuleDict({shape.task: nn.Linear(width, shape.n_classes)})
        self.dataset, self.task = shape.dataset, shape.task

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head[self.task](self.trunk(self.adapter[self.dataset](x)))


class OutsideTheContract(nn.Module):
    input_shape = (784,)

    def __init__(self, shapes, input_shape) -> None:
        super().__init__()
        self.body = nn.Linear(784, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x.flatten(1))


class BareTrunk(External):
    def __init__(self, shapes, input_shape) -> None:
        super().__init__(shapes, input_shape)
        self.trunk = nn.Linear(16, 16)  # keys trunk.weight: not a Sequential


class OwnTensor(External):
    def __init__(self, shapes, input_shape) -> None:
        super().__init__(shapes, input_shape)
        self.scale = nn.Parameter(torch.ones(()))


class Fixed(External):
    def __init__(self, shapes, input_shape) -> None:
        super().__init__(shapes, (1, 28, 28))  # ignores the input it is given


class TenLogits(External):
    def __init__(self, shapes, input_shape) -> None:
        super().__init__(shapes, input_shape)
        self.head = nn.ModuleDict({self.task: nn.Linear(16, 10)})


class MisnamedHead(External):
    def __init__(self, shapes, input_shape) -> None:
        super().__init__(shapes, input_shape)
        self.head = nn.ModuleDict({"letters": self.head[self.task]})


HERE = __name__  # tests.test_learning_torch_module


@pytest.fixture
def trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator trusts this test package, as ONIONFL_TRUSTED_MODULES=tests."""
    monkeypatch.setenv(TRUSTED_ENV, HERE.split(".")[0])


# --- external classes are imported only when trusted, and only at build ---------------

BOOM = "onionfl_boom_245"  # a module that raises if anything imports it


@pytest.fixture
def boom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    (tmp_path / f"{BOOM}.py").write_text(
        "raise RuntimeError('boom: imported')", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv(TRUSTED_ENV, raising=False)
    yield f"{BOOM}:Model"
    sys.modules.pop(BOOM, None)


def test_an_untrusted_reference_is_rejected_without_importing_it(boom: str) -> None:
    with pytest.raises(PluginError, match=f"not trusted.*{TRUSTED_ENV}"):
        models.create("torch_module", {"module": boom})

    assert BOOM not in sys.modules


def test_the_studio_preview_rejects_it_too(boom: str) -> None:
    from onion_fl.studio.previews import sharing_preview

    topology = {"name": "t", "levels": ["global", "edge"], "root": {"id": "cloud"}}
    with pytest.raises(PluginError, match="not trusted"):
        sharing_preview(
            topology, "fedavg", ["d"], {"name": "torch_module", "module": boom}
        )

    assert BOOM not in sys.modules


def test_a_trusted_reference_validates_as_text_and_imports_at_build(
    boom: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TRUSTED_ENV, f"mylab, {BOOM}")
    family = models.create("torch_module", {"module": boom, "input_shape": [784]})

    assert BOOM not in sys.modules  # a valid config imported nothing
    with pytest.raises(RuntimeError, match="boom: imported"):
        family.build([FEMNIST])


@pytest.mark.parametrize(
    "trusted_env, ok",
    [
        ("", False),
        ("tests", True),
        ("tests.test_learning_torch_module", True),
        ("test", False),
        ("tests.other", False),
    ],
)
def test_trust_is_by_module_prefix(
    monkeypatch: pytest.MonkeyPatch, trusted_env: str, ok: bool
) -> None:
    from onion_fl.learning.torch_module import check_reference

    monkeypatch.setenv(TRUSTED_ENV, trusted_env)
    if ok:
        check_reference(f"{HERE}:External")
    else:
        with pytest.raises(TorchModuleError, match="not trusted"):
            check_reference(f"{HERE}:External")
    check_reference("leaf_femnist_cnn")  # presets need no trust


@pytest.mark.parametrize(
    "module, match",
    [
        ("torch.nn:Nope", "cannot import"),
        ("collections:OrderedDict", "not a torch.nn.Module"),
        (f"{HERE}:External", "invalid params"),
    ],
)
def test_a_trusted_class_that_cannot_build_fails_at_build(
    monkeypatch: pytest.MonkeyPatch, module: str, match: str
) -> None:
    monkeypatch.setenv(TRUSTED_ENV, "torch,collections,tests")
    family = models.create(
        "torch_module", {"module": module, "input_shape": [784], "params": {"x": 1}}
    )
    with pytest.raises(TorchModuleError, match=match):
        family.build([FEMNIST])


def test_an_external_class_builds_by_reference_with_its_params(trusted) -> None:
    model = build(f"{HERE}:External", FEMNIST, params={"width": 8})

    assert groups(model) == {"adapter.femnist", "trunk", "head.character"}
    assert model.state_dict()["trunk.1.weight"].shape == (8, 8)
    assert model(torch.zeros((2, 784))).shape == (2, 62)


@pytest.mark.parametrize(
    "cls, match",
    [
        ("OutsideTheContract", "outside adapter/trunk/head"),
        ("BareTrunk", "nn.Sequential"),
        ("OwnTensor", "outside adapter/trunk/head"),
    ],
)
def test_an_external_class_outside_the_contract_is_rejected(
    cls: str, match: str, trusted
):
    with pytest.raises(TorchModuleError, match=match):
        build(f"{HERE}:{cls}", FEMNIST)


def test_a_head_named_after_no_task_is_rejected(trusted) -> None:
    with pytest.raises(TorchModuleError, match="no dataset or task"):
        build(f"{HERE}:MisnamedHead", FEMNIST)


def test_a_preview_without_data_builds_it() -> None:
    from onion_fl.studio.previews import sharing_preview

    topology = {
        "name": "t",
        "levels": ["global", "edge"],
        "root": {"id": "cloud"},
    }
    preview = sharing_preview(
        topology,
        "fedavg",
        ["femnist"],
        {"name": "torch_module", "module": "niid_bench_cnn"},
    )

    assert set(preview["groups"]) == {"adapter.femnist", "trunk", "head.stress"}


# --- hierarchical aggregation equals flat aggregation ---------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_hierarchical_fedavg_equals_flat_fedavg(name: str) -> None:
    # Five edges with the preset's real keys and shapes, two datasets, so the
    # adapters and heads are partial keys; the samples per key are hand-picked.
    _, shape, (dataset, task, classes) = CASES[name]
    other = shape.model_copy(
        update={"dataset": dataset, "task": task, "n_classes": classes}
    )
    config = CASES[name][0]
    edges = [shape, shape, other, shape, other]
    samples = [30, 12, 7, 21, 40]
    updates = []
    for i, (s, n) in enumerate(zip(edges, samples, strict=True)):
        state = state_arrays(build(name, s, seed=i, **config))
        updates.append(
            Contribution(
                source=f"edge_{i}", state=state, weights=dict.fromkeys(state, n)
            )
        )
    fedavg = aggregators.create("fedavg")

    fogs = [
        fedavg.aggregate(updates[:2], source="fog_a"),
        fedavg.aggregate(updates[2:], source="fog_b"),
    ]
    hierarchical = fedavg.aggregate(fogs, source="cloud")
    flat = fedavg.aggregate(updates, source="cloud")

    assert set(flat.state) == set(preset(name, second=True).state_dict())
    assert hierarchical.weights == flat.weights
    trunk = next(k for k in flat.state if k.startswith("trunk."))
    assert flat.weights[trunk] == 110
    for key, value in flat.state.items():
        np.testing.assert_allclose(hierarchical.state[key], value, rtol=1e-5, atol=1e-6)


# --- determinism -------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_same_seed_gives_the_same_weights_and_another_differs(name: str) -> None:
    a, b, c = (state_arrays(preset(name, second=True, seed=s)) for s in (0, 0, 1))

    assert all(np.array_equal(a[k], b[k]) for k in a)
    trunk = next(k for k in a if k.startswith("trunk.") and "weight" in k)
    assert not np.array_equal(a[trunk], c[trunk])


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_part_starts_the_same_in_edge_and_global_models(name: str) -> None:
    edge = state_arrays(preset(name, seed=3))
    global_ = state_arrays(preset(name, second=True, seed=3))

    for key, value in edge.items():
        np.testing.assert_array_equal(value, global_[key])


@pytest.mark.parametrize("name", sorted(CASES))
def test_random_init_restores_every_parameter(name: str) -> None:
    model = preset(name, seed=5)
    before = state_arrays(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    model.init_parameters(5)

    after = state_arrays(model)
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_a_copy_keeps_its_parts_wired_to_the_forward() -> None:
    import copy

    model = preset("niid_bench_cnn").eval()
    twin = copy.deepcopy(model)
    with torch.no_grad():
        for p in twin.head.parameters():
            p.zero_()
    x = torch.zeros((2, 3072))

    assert torch.count_nonzero(twin(x)) == 0
    assert torch.count_nonzero(model(x)) > 0


# --- real data: these tests train ---------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
real = pytest.mark.skipif(
    not (ROOT / "data/FEMNIST/all_data").exists(), reason="data/FEMNIST not available"
)
WRITERS = 12  # of the descriptor's 200: 9 train, 3 test


@pytest.fixture(scope="module")
def femnist_writers():
    from onion_fl.data.ingest import ingest, load_spec

    spec = load_spec(ROOT / "datasets/femnist.yaml")
    return ingest(spec, root=ROOT / "data/FEMNIST")[:WRITERS]


def _writers() -> list[str]:
    from onion_fl.data.ingest import load_spec

    return load_spec(ROOT / "datasets/femnist.yaml").source.reader_params()["writers"]


@real
def test_an_external_class_trains_on_real_femnist(femnist_writers, trusted) -> None:
    from onion_fl.learning.trainers import trainers

    train = femnist_writers[0]
    X = (train.X - train.X.mean()) / train.X.std()
    data = SimpleNamespace(X=X.astype(np.float32), y=train.y)
    model = family(f"{HERE}:External").build([train.shape], seed=0)
    trainer = trainers.create("standard", {"local_epochs": 1, "lr": 0.01})
    ctx = SimpleNamespace(rng=np.random.default_rng(0))

    first = trainer.train(model, data, ctx=ctx)
    for _ in range(4):
        last = trainer.train(model, data, ctx=ctx)

    assert last.loss < first.loss


@real
def test_the_leaf_cnn_learns_femnist_end_to_end(tmp_path: Path, femnist_writers):
    # 9 real writers train over two fogs and 3 others test; the preprocessing
    # drops the constant pixels, so the CNN sees images rebuilt from x_<i>.
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import run_scenario
    from onion_fl.experiment.sweep import scenarios
    from onion_fl.observability.run import verify_run

    names = [s.subject for s in femnist_writers]
    link = {"transport": "memory", "codec": "npz", "profile": "lan"}
    config = {
        "name": "femnist_cnn_e2e",
        "topology": {
            "name": "two_fogs",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {
                "defaults": {"link_up": link},
                "nodes": [{"id": "fog_a"}, {"id": "fog_b"}],
            },
            "edge": {"link_up": link},
        },
        "data": {
            "datasets": {"femnist": {}},
            "roles": {
                "seed": 0,
                "overrides": {
                    "femnist": {
                        "test": names[-3:],
                        "exclude": [w for w in _writers() if w not in names],
                    }
                },
            },
            "placement": {"name": "mixing", "alpha": 1.0},
        },
        "learning": {
            "model": {"name": "torch_module", "module": "leaf_femnist_cnn"},
            "trainer": {"name": "standard", "local_epochs": 1, "lr": 0.001},
        },
        "rounds": 3,
        "evaluation": {"metrics": ["loss", "accuracy"], "global": {"every": 1}},
        "seeds": [0],
        "paths": {
            "datasets": str(ROOT / "datasets"),
            "runs": str(tmp_path / "runs"),
            "cache": str(tmp_path / "cache"),
        },
    }
    (scenario,) = scenarios(parse_experiment(config))

    path = run_scenario(scenario)

    assert verify_run(path)
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    final = json.loads((path / "summary.json").read_text(encoding="utf-8"))["final"]
    # the training writers' most frequent class, scored on the test writers
    train = np.concatenate([s.y for s in femnist_writers[:-3]])
    test = np.concatenate([s.y for s in femnist_writers[-3:]])
    majority = float(np.mean(test == np.bincount(train).argmax()))
    assert final["global/femnist"]["accuracy"] > 2 * majority
