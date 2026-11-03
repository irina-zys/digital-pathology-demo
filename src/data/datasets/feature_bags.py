"""Complete frozen-feature bags."""
from torch.utils.data import Dataset


class FeatureBagDataset(Dataset):
    def __init__(self, samples, arrays):
        self.samples, self.arrays = samples, arrays

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index][0]
        return {'features': self.arrays[row['bag_id']], 'target': int(row['label']), 'id': row['bag_id']}
