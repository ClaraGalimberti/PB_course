import torch
from plants.custom_dataset import CustomDataset


class TankDataset(CustomDataset):
    def __init__(self, random_seed, horizon, std_ini=0.2, x0=None):
        # experiment and file names
        exp_name = 'tank'
        file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_RS' + str(random_seed) + '.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.state_dim = 1

        if x0 is None:
            self.x0 = torch.zeros(self.state_dim)
        else:
            self.x0 = x0

    # ---- data generation ----
    def _generate_data(self, num_samples):
        data = self.std_ini * torch.randn(num_samples, self.horizon, self.state_dim)
        for rollout_num in range(num_samples):
            data[rollout_num, 0, :] = self.x0 + data[rollout_num, 0, :]
        assert data.shape[0] == num_samples
        return data
