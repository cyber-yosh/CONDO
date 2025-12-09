# Adapted from: MoCo (Facebook AI Research)
# https://github.com/facebookresearch/moco


class TwoCropsTransform:
    """Take two rando20augmentations of one image."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class DistributedBalancedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.replacement = replacement
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.targets = torch.tensor(dataset.targets)
        class_sample_count = torch.tensor(
            [(self.targets == t).sum() for t in torch.unique(self.targets)]
        )
        self.weights = torch.tensor([1.0 / class_sample_count[t] for t in self.targets])

        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator().manual_seed(self.epoch)

        sampled_indices = torch.multinomial(
            self.weights, self.total_size, self.replacement
        )

        # Subsample for this rank
        per_rank_indices = sampled_indices[self.rank :: self.num_replicas]

        return iter(per_rank_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
