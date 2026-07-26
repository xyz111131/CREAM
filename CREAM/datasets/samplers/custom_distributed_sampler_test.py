from torch.utils.data import Dataset
import unittest
import numpy as np

from CREAM.datasets.samplers.custom_distributed_sampler import CustomDistributedSampler


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = list(range(size))

        self.genomic_regions_df = np.zeros([size, 2000])
        self.num_individuals_per_gene = 128
        self.n_gene_replicate_batches_per_epoch = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestCustomDistributedSampler(unittest.TestCase):
    def test_indices_per_rank(self):
        dataset = DummyDataset(size=100)
        num_replicas = 2
        seed = 123
        epoch = 0

        for rank in range(num_replicas):
            sampler = CustomDistributedSampler(
                dataset=dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            sampler.set_epoch(epoch)
            indices = list(iter(sampler))
            print(f"Rank {rank} indices: {indices}")
            self.assertTrue(len(indices) in [3, 4])  # 10 items split among 3


if __name__ == "__main__":
    unittest.main()
