from __future__ import annotations

"""Subject roles, clients and train-only preprocessing (spec §8.4).

Roles are drawn per dataset with their own seeded stream, before and apart
from the placement: the test subjects are the same in every scenario that
shares the seed, and adding a dataset does not move another one's roles.

==========  ===============================================================
``test``    reserved subjects; global evaluators, identical across scenarios
``val``     reserved subjects the placement turns into zone evaluators
``train``   the bag the placement distributes, grouped into clients
==========  ===============================================================

Imputation, scaling and the removal of constant features are fitted on the
training portions only: never on val, test or the ``local_val`` rows. With
``scaler: local`` every data holder (client or evaluator) standardises with
its own statistics, which use no labels.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from onion_fl.core.context import node_rng
from onion_fl.data.contract import DataError, SubjectData, natural_key

Share = Annotated[float, Field(ge=0, lt=1)]


class RoleOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test: Share | list[str] | None = None
    val: Share | list[str] | None = None
    exclude: list[str] = Field(
        default_factory=list, description="Sujetos que no toman ningún rol"
    )


class RolesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test: Share = Field(
        0.2, description="Proporción de sujetos reservados para el test"
    )
    val: Share = Field(
        0.0, description="Proporción de sujetos para evaluadores de zona"
    )
    overrides: dict[str, RoleOverride] = Field(
        default_factory=dict,
        description="Por dataset: proporciones o listas de sujetos",
    )
    seed: int = 0
    local_val: Share = Field(
        0.0, description="Cola de cada sujeto de entrenamiento para validación local"
    )
    local_val_split: Literal["tail", "class_tail"] = Field(
        "tail",
        description="tail: últimas filas del sujeto; class_tail: últimas filas de cada clase",
        exclude_if=lambda v: v == "tail",  # unset, it keeps existing config_ids
    )
    subjects_per_client: PositiveInt = 1
    scaler: Literal["global", "local", "none"] = Field(
        "global", description="global: por dataset; local: cada nodo con sus datos"
    )
    impute: Literal["mean", "median"] = "mean"
    drop_constant: bool = Field(
        True, description="Quitar features constantes en la bolsa de entrenamiento"
    )


@dataclass(frozen=True)
class Client:
    """One edge's data: one or more training subjects of one dataset."""

    id: str
    dataset: str
    subjects: tuple[str, ...]
    train: SubjectData
    local_val: SubjectData | None


@dataclass(frozen=True)
class DataSplit:
    clients: list[Client]
    val: list[SubjectData]
    test: list[SubjectData]
    roles: dict[str, dict[str, list[str]]]  # dataset -> role -> subjects
    # dataset -> kept features, fill, mean, std: JSON-ready, frozen by a continuation
    preprocessing: dict[str, dict[str, Any]] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "roles": self.roles,
            "clients": [
                {
                    "id": c.id,
                    "dataset": c.dataset,
                    "subjects": list(c.subjects),
                    "samples": c.train.n_samples,
                    "class_counts": c.train.class_counts,
                    "local_val_samples": c.local_val.n_samples if c.local_val else 0,
                }
                for c in self.clients
            ],
        }


@dataclass(frozen=True)
class _Fit:
    """Fill values, mean and std fitted on some rows (std 0 is kept as 1)."""

    fill: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def on(cls, X: np.ndarray, impute: str, fallback: np.ndarray | None = None) -> _Fit:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            reduce = np.nanmedian if impute == "median" else np.nanmean
            fill = reduce(X, axis=0) if len(X) else np.full(X.shape[1], np.nan)
        if fallback is not None:
            fill = np.where(np.isnan(fill), fallback, fill)
        fill = np.nan_to_num(fill)
        filled = np.where(np.isnan(X), fill, X)
        if not len(X):
            return cls(fill, np.zeros_like(fill), np.ones_like(fill))
        std = filled.std(axis=0)
        return cls(fill, filled.mean(axis=0), np.where(std > 0, std, 1.0))

    def take(self, keep: np.ndarray) -> _Fit:
        return _Fit(self.fill[keep], self.mean[keep], self.std[keep])

    def apply(self, X: np.ndarray, scale: bool) -> np.ndarray:
        filled = np.where(np.isnan(X), self.fill, X)
        return ((filled - self.mean) / self.std if scale else filled).astype(np.float32)


def _pick(
    share: float | list[str], order: list[str], taken: set[str], role: str, ds: str
) -> list[str]:
    if isinstance(share, list):
        unknown = sorted(set(share) - set(order), key=natural_key)
        if unknown:
            raise DataError(f"{ds}: {role} subjects {unknown} do not exist")
        both = sorted(set(share) & taken, key=natural_key)
        if both:
            raise DataError(f"{ds}: subjects {both} are in both test and val")
        return sorted(share, key=natural_key)
    n = 0 if share == 0 else max(1, round(share * len(order)))
    return sorted([s for s in order if s not in taken][:n], key=natural_key)


def _assign(
    names: list[str], dataset: str, config: RolesConfig
) -> dict[str, list[str]]:
    override = config.overrides.get(dataset, RoleOverride())
    excluded = sorted(set(override.exclude), key=natural_key)
    unknown = [s for s in excluded if s not in names]
    if unknown:
        raise DataError(f"{dataset}: excluded subjects {unknown} do not exist")
    for role in ("test", "val"):
        listed = getattr(override, role)
        clash = (
            sorted(set(listed) & set(excluded), key=natural_key)
            if isinstance(listed, list)
            else []
        )
        if clash:
            raise DataError(f"{dataset}: subjects {clash} are both {role} and excluded")
    rng = node_rng(config.seed, f"roles/{dataset}")
    # Draw over every subject, then drop the excluded: the others keep their order.
    order = [names[i] for i in rng.permutation(len(names)) if names[i] not in excluded]
    test_share = config.test if override.test is None else override.test
    val_share = config.val if override.val is None else override.val
    test = _pick(test_share, order, set(), "test", dataset)
    val = _pick(val_share, order, set(test), "val", dataset)
    train = [s for s in names if s not in set(test) | set(val) | set(excluded)]
    if not train:
        raise DataError(f"{dataset}: no training subjects left after test and val")
    return {"test": test, "val": val, "train": train, "excluded": excluded}


def _like(data: SubjectData, X, y, subject: str, names: list[str]) -> SubjectData:
    return SubjectData(
        X=X,
        y=y,
        dataset=data.dataset,
        subject=subject,
        task=data.task,
        n_classes=data.n_classes,
        feature_names=names,
    )


def _held_out(y: np.ndarray, share: float, split: str) -> np.ndarray:
    """Rows of one subject kept for its local validation, in time order.

    ``tail`` takes the last rows, which in a recording usually hold a single
    condition; ``class_tail`` takes the last rows of each class instead.
    """
    n = len(y)
    held = np.zeros(n, dtype=bool)
    if split == "tail":
        held[n - min(round(share * n), n - 1) :] = True
        return held
    for label in np.unique(y):
        rows = np.flatnonzero(y == label)
        k = round(share * len(rows))
        if k:
            held[rows[-k:]] = True
    if held.all():
        held[0] = False  # a subject always keeps a training row
    return held


def split_subjects(
    subjects: Sequence[SubjectData],
    config: RolesConfig | None = None,
    frozen: Mapping[str, Mapping[str, Any]] | None = None,
) -> DataSplit:
    """Assign roles per dataset, group the training subjects into clients and preprocess.

    A dataset in ``frozen`` (a parent run's ``preprocessing``) keeps those features
    and statistics instead of fitting new ones; the other datasets are fitted.
    """
    config = config or RolesConfig()
    by_dataset: dict[str, dict[str, SubjectData]] = {}
    for data in subjects:
        by_dataset.setdefault(data.dataset, {})[data.subject] = data
    unknown = sorted(set(config.overrides) - set(by_dataset))
    if unknown:
        raise DataError(f"overrides for datasets {unknown} that are not loaded")

    scale, local = config.scaler != "none", config.scaler == "local"
    clients, val, test, roles = [], [], [], {}
    preprocessing: dict[str, dict[str, Any]] = {}
    for dataset in sorted(by_dataset):
        pool = by_dataset[dataset]
        names = sorted(pool, key=natural_key)
        roles[dataset] = _assign(names, dataset, config)

        # Rows each training subject keeps for its local validation.
        held = {
            name: _held_out(pool[name].y, config.local_val, config.local_val_split)
            for name in roles[dataset]["train"]
        }
        bag = np.concatenate([pool[s].X[~held[s]] for s in held])

        own = pool[names[0]].feature_names
        if frozen is not None and dataset in frozen:
            saved = dict(frozen[dataset])
            missing = [f for f in saved["features"] if f not in own]
            if missing:
                raise DataError(
                    f"{dataset}: the frozen preprocessing needs features {missing} "
                    "that the data lacks"
                )
            if saved["scaler"] != config.scaler:
                raise DataError(
                    f"{dataset}: the frozen preprocessing scales {saved['scaler']!r}, "
                    f"not {config.scaler!r}"
                )
            keep = np.array([own.index(f) for f in saved["features"]])
            shared = _Fit(*(np.asarray(saved[k]) for k in ("fill", "mean", "std")))
        else:
            fit = _Fit.on(bag, config.impute)
            keep = np.arange(bag.shape[1])
            if config.drop_constant:
                keep = np.flatnonzero(
                    np.where(np.isnan(bag), fit.fill, bag).std(axis=0) > 0
                )
                if not len(keep):
                    raise DataError(
                        f"{dataset}: every feature is constant in the training bag"
                    )
            shared = fit.take(keep)
            saved = {
                "scaler": config.scaler,
                "impute": config.impute,
                "features": [own[i] for i in keep],
                **{k: getattr(shared, k).tolist() for k in ("fill", "mean", "std")},
            }
        preprocessing[dataset] = saved
        feature_names = [own[i] for i in keep]

        def holder_fit(X: np.ndarray, shared: _Fit = shared) -> _Fit:
            return _Fit.on(X, config.impute, fallback=shared.fill) if local else shared

        for role, out in (("val", val), ("test", test)):
            for name in roles[dataset][role]:
                X = pool[name].X[:, keep]
                out.append(
                    _like(
                        pool[name],
                        holder_fit(X).apply(X, scale),
                        pool[name].y,
                        name,
                        feature_names,
                    )
                )

        rng = node_rng(config.seed, f"clients/{dataset}")
        order = [roles[dataset]["train"][i] for i in rng.permutation(len(held))]
        for start in range(0, len(order), config.subjects_per_client):
            group = tuple(
                sorted(
                    order[start : start + config.subjects_per_client], key=natural_key
                )
            )
            client_id = "-".join([dataset, *group])
            rows = [pool[s] for s in group]
            train_X = np.concatenate([d.X[~held[d.subject]][:, keep] for d in rows])
            train_y = np.concatenate([d.y[~held[d.subject]] for d in rows])
            tail_X = np.concatenate([d.X[held[d.subject]][:, keep] for d in rows])
            tail_y = np.concatenate([d.y[held[d.subject]] for d in rows])
            own = holder_fit(train_X)
            clients.append(
                Client(
                    id=client_id,
                    dataset=dataset,
                    subjects=group,
                    train=_like(
                        rows[0],
                        own.apply(train_X, scale),
                        train_y,
                        client_id,
                        feature_names,
                    ),
                    local_val=(
                        _like(
                            rows[0],
                            own.apply(tail_X, scale),
                            tail_y,
                            client_id,
                            feature_names,
                        )
                        if len(tail_y)
                        else None
                    ),
                )
            )
    clients.sort(key=lambda c: (c.dataset, natural_key(c.id)))
    return DataSplit(
        clients=clients, val=val, test=test, roles=roles, preprocessing=preprocessing
    )
