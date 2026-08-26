"""
Tests for the v2 data preparation pipeline: filtering, splitting, scaling, record building.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from estimint.v2.data.features import FEATURES_BASE, StandardScaler, get_features
from estimint.v2.data.preprocess import (
    SplitParamSims,
    _assign_param_group,
    _assign_param_splits,
    _build_data,
    _create_split,
    _filter_by_threshold,
    _fit_features_scaler,
    _fit_target_scaler,
    _load_split,
    _parameter_strata,
    _save_split,
    prepare_data,
)

N_PARAMS = 20
N_SIMS = 2
ROWS_PER_SIM = 3


def make_df(n_params=N_PARAMS, n_sims=N_SIMS, rows_per_sim=ROWS_PER_SIM, low_prev_params=()):
    """Synthetic simulation frame: constant feature/target values within a (param, sim) pair."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(n_params):
        for s in range(n_sims):
            static = {f: float(rng.uniform(0, 1)) for f in FEATURES_BASE}
            static["prev_y9"] = 0.001 if p in low_prev_params else float(rng.uniform(0.05, 0.8))
            static["hbr_y9"] = float(rng.uniform(1, 100))
            static["eir"] = float(rng.uniform(0.1, 500))
            rows += [{"parameter_index": p, "simulation_index": s, **static}] * rows_per_sim
    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return make_df()


@pytest.fixture
def filtered_df(df):
    return _filter_by_threshold(df)


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "seed": 0,
            "predictor": "prev_y9",
            "target": "eir",
            "stratify": False,
            "use_existing_split": False,
            "split_file": str(tmp_path / "split.csv"),
            "output_dir": str(tmp_path / "out"),
        }
    )


def param_indices(pairs):
    return {p for p, _ in pairs}


class TestFilterByThreshold:
    def test_drops_pairs_below_prevalence_threshold(self):
        df = make_df(n_params=6, low_prev_params=(1, 4))
        out = _filter_by_threshold(df)
        assert param_indices(out["_ps"]) == {0, 2, 3, 5}

    def test_adds_pair_column(self, filtered_df):
        assert list(filtered_df["_ps"].iloc[0]) == [
            filtered_df["parameter_index"].iloc[0],
            filtered_df["simulation_index"].iloc[0],
        ]

    def test_keeps_everything_above_threshold(self, df, filtered_df):
        assert len(filtered_df) == len(df)


class TestSplitting:
    def test_splits_are_disjoint_and_cover_all_pairs(self, filtered_df):
        split = _create_split(filtered_df, seed=0, calib_frac=0.1)
        all_pairs = set(filtered_df["_ps"])
        parts = [split.train, split.val, split.calib, split.test]
        assert set().union(*parts) == all_pairs
        assert sum(len(p) for p in parts) == len(all_pairs)

    def test_a_parameter_never_straddles_two_splits(self, filtered_df):
        split = _create_split(filtered_df, seed=0, calib_frac=0.1)
        groups = [param_indices(p) for p in (split.train, split.val, split.calib, split.test)]
        for i, a in enumerate(groups):
            for b in groups[i + 1 :]:
                assert not (a & b)

    def test_split_is_deterministic_given_a_seed(self, filtered_df):
        assert _create_split(filtered_df, seed=0).train == _create_split(filtered_df, seed=0).train

    def test_different_seeds_give_different_splits(self, filtered_df):
        assert _create_split(filtered_df, seed=0).train != _create_split(filtered_df, seed=1).train

    def test_train_gets_roughly_seventy_percent(self, filtered_df):
        split = _create_split(filtered_df, seed=0, stratify=False)
        assert len(param_indices(split.train)) == pytest.approx(0.7 * N_PARAMS, abs=1)

    def test_calibration_split_is_empty_by_default(self, filtered_df):
        assert _create_split(filtered_df, seed=0).calib == set()

    def test_rejects_calibration_fraction_that_starves_validation(self, filtered_df):
        with pytest.raises(ValueError, match="Validation fraction"):
            _assign_param_splits(filtered_df, seed=0, calib_frac=0.4)

    def test_assign_param_group_respects_fractions(self):
        assign = _assign_param_group(
            np.arange(100), np.random.default_rng(0), train_frac=0.7, val_frac=0.15, calib_frac=0.05
        )
        counts = pd.Series(list(assign.values())).value_counts()
        assert counts["train"] == 70 and counts["val"] == 15 and counts["calib"] == 5 and counts["test"] == 10


class TestParameterStrata:
    def test_unstratified_returns_one_group(self, filtered_df):
        strata = _parameter_strata(filtered_df, "eir", n_bins=10, stratify=False)
        assert len(strata) == 1
        assert len(strata[0]) == N_PARAMS

    def test_stratified_partitions_every_parameter_once(self, filtered_df):
        strata = _parameter_strata(filtered_df, "eir", n_bins=5, stratify=True)
        assert len(strata) <= 5
        assert sorted(np.concatenate(strata)) == sorted(filtered_df["parameter_index"].unique())

    def test_strata_are_ordered_by_target_magnitude(self, filtered_df):
        strata = _parameter_strata(filtered_df, "eir", n_bins=4, stratify=True)
        means = filtered_df.groupby("parameter_index")["eir"].mean()
        stratum_means = [means[s].mean() for s in strata]
        assert stratum_means == sorted(stratum_means)


class TestSplitFileRoundTrip:
    def test_save_then_load_preserves_splits(self, filtered_df, tmp_path):
        split = _create_split(filtered_df, seed=0, calib_frac=0.1)
        path = tmp_path / "split.csv"
        _save_split(path, split)
        loaded = _load_split(str(path), filtered_df)
        assert loaded == split

    def test_load_ignores_pairs_absent_from_the_frame(self, filtered_df, tmp_path):
        path = tmp_path / "split.csv"
        split = _create_split(filtered_df, seed=0)
        _save_split(path, split)
        smaller = filtered_df[filtered_df["parameter_index"] < 5]
        loaded = _load_split(str(path), smaller)
        assert param_indices(loaded.train | loaded.val | loaded.test) <= {0, 1, 2, 3, 4}

    def test_load_maps_csv_split_names(self, filtered_df, tmp_path):
        path = tmp_path / "split.csv"
        _save_split(path, SplitParamSims(train={(0, 0)}, val={(1, 0)}, calib={(2, 0)}, test={(3, 0)}))
        assert set(pd.read_csv(path)["split"]) == {"train", "validate", "calibrate", "test"}
        loaded = _load_split(str(path), filtered_df)
        assert loaded.val == {(1, 0)} and loaded.calib == {(2, 0)}


class TestScalers:
    def test_feature_scaler_uses_train_pairs_only(self, filtered_df, tmp_path):
        train_ps = {ps for ps in filtered_df["_ps"] if ps[0] < 5}
        features = get_features("prev_y9")
        scaler = _fit_features_scaler(filtered_df, train_ps, str(tmp_path), features)

        expected = (
            filtered_df[filtered_df["_ps"].isin(train_ps)]
            .drop_duplicates(subset=["_ps"])[features]
            .to_numpy(dtype=np.float32)
        )
        np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0), rtol=1e-5)

    def test_feature_scaler_is_pickled_to_the_output_dir(self, filtered_df, tmp_path):
        train_ps = set(filtered_df["_ps"])
        scaler = _fit_features_scaler(filtered_df, train_ps, str(tmp_path), get_features("prev_y9"))
        with open(tmp_path / "features_scaler.pkl", "rb") as f:
            np.testing.assert_allclose(pickle.load(f).mean_, scaler.mean_)

    def test_target_scaler_is_fitted_in_log_space(self, filtered_df, tmp_path):
        train_ps = set(filtered_df["_ps"])
        scaler = _fit_target_scaler(filtered_df, train_ps, str(tmp_path), target="eir")

        eir = filtered_df.drop_duplicates(subset=["_ps"])["eir"].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(scaler.mean_, np.log10(eir).mean(), rtol=1e-4)
        assert (tmp_path / "target_scaler.pkl").exists()

    def test_wide_range_predictor_is_fitted_in_log_space(self, filtered_df, tmp_path):
        # eir spans ~3 decades, so it is standardized in log space like the target.
        # The bounded covariates in FEATURES_BASE are left alone.
        features = get_features("eir")
        scaler = _fit_features_scaler(filtered_df, set(filtered_df["_ps"]), str(tmp_path), features)

        static = filtered_df.drop_duplicates(subset=["_ps"])
        eir = static["eir"].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(scaler.mean_[0], np.log10(eir).mean(), rtol=1e-4)
        np.testing.assert_allclose(
            scaler.mean_[1:], static[FEATURES_BASE].to_numpy(dtype=np.float32).mean(axis=0), rtol=1e-5
        )


class TestBuildData:
    @pytest.fixture
    def records(self, filtered_df, tmp_path):
        filtered_df = filtered_df.assign(_weight=1.0)
        features = get_features("prev_y9")
        pairs = set(filtered_df["_ps"])
        feature_scaler = _fit_features_scaler(filtered_df, pairs, str(tmp_path), features)
        target_scaler = _fit_target_scaler(filtered_df, pairs, str(tmp_path), "eir")
        data = _build_data(filtered_df, pairs, feature_scaler, target_scaler, features, "eir")
        return data, feature_scaler, target_scaler, features

    def test_one_record_per_pair_with_expected_keys(self, records, filtered_df):
        data, *_ = records
        assert len(data) == len(set(filtered_df["_ps"]))
        assert set(data[0]) == {"x_raw", "x", "y_raw", "y", "y_std", "w", "ps"}

    def test_features_are_scaled_versions_of_the_raw_row(self, records):
        data, feature_scaler, _, features = records
        record = data[0]
        assert record["x_raw"].shape == (len(features),)
        np.testing.assert_allclose(record["x"], feature_scaler.transform(record["x_raw"]), rtol=1e-5)

    def test_target_is_log10_then_standardized(self, records):
        data, _, target_scaler, _ = records
        record = data[0]
        assert record["y"] == pytest.approx(np.log10(record["y_raw"]), rel=1e-6)
        assert record["y_std"] == pytest.approx(
            target_scaler.transform(np.array([[record["y"]]], dtype=np.float32))[0, 0], rel=1e-5
        )

    def test_unknown_pairs_are_skipped(self, filtered_df, tmp_path):
        filtered_df = filtered_df.assign(_weight=1.0)
        features = get_features("prev_y9")
        scaler = _fit_features_scaler(filtered_df, set(filtered_df["_ps"]), str(tmp_path), features)
        target_scaler = _fit_target_scaler(filtered_df, set(filtered_df["_ps"]), str(tmp_path), "eir")
        data = _build_data(filtered_df, {(0, 0), (999, 999)}, scaler, target_scaler, features, "eir")
        assert len(data) == 1


class TestPrepareData:
    def test_end_to_end(self, df, cfg):
        prepared = prepare_data(df, cfg, calib_frac=0.1)

        assert prepared.input_size == len(get_features(cfg.predictor))
        assert all(len(d) > 0 for d in (prepared.train_data, prepared.val_data, prepared.calib_data, prepared.test_data))
        n_records = sum(len(d) for d in (prepared.train_data, prepared.val_data, prepared.calib_data, prepared.test_data))
        assert n_records == N_PARAMS * N_SIMS
        assert isinstance(prepared.feature_scaler, StandardScaler) and prepared.feature_scaler.is_fitted

    def test_writes_split_and_scalers(self, df, cfg, tmp_path):
        prepare_data(df, cfg, calib_frac=0.1)
        assert (tmp_path / "split.csv").exists()
        assert (tmp_path / "out" / "features_scaler.pkl").exists()
        assert (tmp_path / "out" / "target_scaler.pkl").exists()

    def test_reuses_an_existing_split_file(self, df, cfg):
        first = prepare_data(df, cfg, calib_frac=0.1)
        cfg.use_existing_split = True
        cfg.seed = 999  # would produce a different split if it were recreated
        second = prepare_data(df, cfg, calib_frac=0.1)
        assert second.train_param_sims == first.train_param_sims

    def test_low_prevalence_pairs_are_excluded(self, cfg):
        prepared = prepare_data(make_df(low_prev_params=(0, 1)), cfg, calib_frac=0.1)
        kept = set().union(
            prepared.train_param_sims, prepared.val_param_sims, prepared.calib_param_sims, prepared.test_param_sims
        )
        assert not param_indices(kept) & {0, 1}

    def test_scalers_ignore_non_train_pairs(self, df, cfg):
        prepared = prepare_data(df, cfg, calib_frac=0.1)
        train_x = np.stack([r["x_raw"] for r in prepared.train_data])
        np.testing.assert_allclose(prepared.feature_scaler.mean_, train_x.mean(axis=0), rtol=1e-4)

    def test_feature_scaler_reproduces_every_record_from_its_raw_row(self, df, cfg):
        # Training consumes record["x"], while compute_metrics and RQSArtifact re-derive
        # the context from record["x_raw"] via prepared.feature_scaler. Any feature
        # transform has to live inside the scaler or those two paths drift apart.
        cfg.predictor = "hbr_y9"  # wide-range predictor, so a log transform is in play
        prepared = prepare_data(df, cfg, calib_frac=0.1)

        for split in (prepared.train_data, prepared.val_data, prepared.calib_data, prepared.test_data):
            x_raw = np.stack([r["x_raw"] for r in split])
            x = np.stack([r["x"] for r in split])
            np.testing.assert_allclose(x, prepared.feature_scaler.transform(x_raw), rtol=1e-5)
