"""
Tests for the Grain data source and loader.
"""

import numpy as np
import pytest

from estimint.v2.data.dataset import DataSource, make_loader


def make_records(n=10, n_features=3):
    return [
        {
            "x": np.full(n_features, i, dtype=np.float32),
            "y_std": np.float32(i),
            "w": np.float32(1.0),
            "ps": np.array([i, 0], dtype=np.int32),
        }
        for i in range(n)
    ]


@pytest.fixture
def records():
    return make_records()


class TestDataSource:
    def test_length_matches_the_records(self, records):
        assert len(DataSource(records)) == len(records)

    def test_getitem_returns_the_record(self, records):
        assert DataSource(records)[3] is records[3]


class TestMakeLoader:
    def test_batches_have_a_leading_batch_dimension(self, records):
        batch = next(iter(make_loader(records, batch_size=5)))
        assert batch["x"].shape == (5, 3)
        assert batch["y_std"].shape == (5,)
        assert batch["ps"].shape == (5, 2)

    def test_drop_remainder_discards_the_partial_batch(self, records):
        batches = list(make_loader(records, batch_size=4, drop_remainder=True))
        assert [b["y_std"].shape[0] for b in batches] == [4, 4]

    def test_keeping_the_remainder_yields_every_record(self, records):
        batches = list(make_loader(records, batch_size=4, drop_remainder=False))
        assert [b["y_std"].shape[0] for b in batches] == [4, 4, 2]

    def test_unshuffled_loader_preserves_record_order(self, records):
        batches = list(make_loader(records, batch_size=5, shuffle=False))
        seen = np.concatenate([b["y_std"] for b in batches])
        np.testing.assert_array_equal(seen, np.arange(len(records), dtype=np.float32))

    def test_shuffle_reorders_records_without_losing_any(self, records):
        batches = list(make_loader(records, batch_size=5, shuffle=True, seed=0))
        seen = np.concatenate([b["y_std"] for b in batches])
        assert sorted(seen) == list(np.arange(len(records), dtype=np.float32))

    def test_same_seed_gives_the_same_shuffle(self, records):
        order = lambda seed: np.concatenate(
            [b["y_std"] for b in make_loader(records, batch_size=5, shuffle=True, seed=seed)]
        )
        np.testing.assert_array_equal(order(0), order(0))
        assert not np.array_equal(order(0), order(1))

    def test_loader_covers_exactly_one_epoch(self, records):
        assert sum(b["y_std"].shape[0] for b in make_loader(records, batch_size=2)) == len(records)

    def test_full_batch_loads_everything_at_once(self, records):
        batches = list(make_loader(records, batch_size=len(records)))
        assert len(batches) == 1 and batches[0]["x"].shape == (len(records), 3)
