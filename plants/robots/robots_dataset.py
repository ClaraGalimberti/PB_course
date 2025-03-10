import torch
from plants.custom_dataset import CustomDataset


class RobotsDataset(CustomDataset):
    def __init__(self, random_seed, horizon, std_ini=0.2):
        # experiment and file names
        exp_name = 'robot'
        file_name = 'data_T' + str(horizon) + '_stdini' + str(std_ini) + '_RS' + str(
            random_seed) + '.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini


    # ---- data generation ----
    def _generate_data(self, num_samples):
        state_dim = 4
        data = torch.zeros(num_samples, self.horizon, state_dim)
        for rollout_num in range(num_samples):
            data[rollout_num, 0, :] = self.std_ini * torch.randn(state_dim)

        assert data.shape[0] == num_samples
        return data
