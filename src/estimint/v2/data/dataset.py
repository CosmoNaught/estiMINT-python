import grain.python as grain


class DataSource(grain.RandomAccessDataSource):
    def __init__(self, data: list[dict]):
        """
        Store records for random access.

        Args:
            data: Sequence records.

        Returns:
            None.
        """
        self.data = data

    def __len__(self) -> int:
        """
        Return the number of records.

        Returns:
            Number of records.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one record by index.

        Args:
            idx: Record index.

        Returns:
            Record dictionary.
        """
        return self.data[idx]


def make_loader(
    data: list[dict],
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    drop_remainder: bool = True,
) -> grain.DataLoader:
    """
    Build a Grain data loader.

    Args:
        data: Sequence records.
        batch_size: Batch size.
        shuffle: Whether to shuffle indices.
        seed: Sampler seed.
        num_workers: Worker count.
        drop_remainder: Whether to drop partial batches.

    Returns:
        Configured Grain data loader.
    """
    data_source = DataSource(data)

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=shuffle,
        seed=seed,
    )

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=num_workers,
    )
    return loader
