import torch
import torch.nn as nn
import torch.nn.functional as F


class ConDo(nn.Module):
    """

    Args:
        encoder: The encoder network.
        d_cat: The dimension of the categorical (class) part of the embedding.
        d_inst: The dimension of the instance (other) part of the embedding.
        mlp: Whether to use a multi-layer perceptron (MLP) head.
        temperature: The temperature parameter for the contrastive loss.
        class_weight: The weight for the classification loss.
        eps: A small value to avoid division by zero.
        logit_temp: The temperature parameter for the classification logits.
    """

    def __init__(
        self,
        encoder,
        dim_in=2048,
        d_cat=10,
        d_inst=128,
        mlp=True,
        temperature=0.1,
        class_weight=5.0,
        eps=1e-8,
        logit_temp=0.1,
    ):
        super(ConDo, self).__init__()
        self.encoder = encoder
        self.dim_in = dim_in
        self.mlp = mlp
        self.d_cat = d_cat
        self.d_inst = d_inst
        self.temperature = temperature
        self.class_weight = class_weight
        self.eps = eps
        self.logit_temp = logit_temp

        # dim_in = self.encoder.fc.weight.shape[1]

        # Remove the encoder's final classification layer
        self.encoder.fc = nn.Identity()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.1)

        if self.mlp:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.dim_in, 1536),
                nn.BatchNorm1d(1536),
                nn.ReLU(),
                nn.Linear(1536, d_cat + d_inst),
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(self.dim_in, d_cat + d_inst)
            )

    def forward(self, img_0, img_1, labels):
        out0 = self.head(self.encoder(img_0))
        out1 = self.head(self.encoder(img_1))

        # Split and normalize embeddings
        z_c0 = F.normalize(out0[:, : self.d_cat], dim=1)
        z_n0 = F.normalize(out0[:, self.d_cat :], dim=1)
        z_c1 = F.normalize(out1[:, : self.d_cat], dim=1)
        z_n1 = F.normalize(out1[:, self.d_cat :], dim=1)

        z_c = torch.cat([z_c0, z_c1], dim=0)
        z_n = torch.cat([z_n0, z_n1], dim=0)
        labels_2b = torch.cat([labels, labels], dim=0)

        # Compute similarity matrices
        sim_c = (z_c @ z_c.T) / self.temperature
        sim_n = (z_n @ z_n.T) / self.temperature

        # Create positive mask (exclude self-comparisons)
        mask_pos = (labels_2b.unsqueeze(0) == labels_2b.unsqueeze(1)) & ~torch.eye(
            labels_2b.shape[0], dtype=torch.bool, device=labels_2b.device
        )

        sim_c_stable = sim_c - sim_c.max(dim=1, keepdim=True)[0].detach()
        exp_sim_c = torch.exp(sim_c_stable)

        # Compute negative weight
        diag_sim_n = torch.diag(sim_n)
        exp_factor = torch.exp(sim_n - diag_sim_n.unsqueeze(1))

        # Compute negative weight
        inner_sum = (exp_factor * exp_sim_c).sum(dim=1, keepdim=True)  # (2B, 1)

        # Full denominator
        denominator = inner_sum + exp_sim_c

        # Log probability
        log_probs = sim_c_stable - torch.log(denominator + self.eps)

        # Compute contrastive loss
        masked_log_probs = log_probs * mask_pos.float()
        num_pos_per_anchor = mask_pos.sum(dim=1).float()
        valid_anchors = num_pos_per_anchor > 0

        if not valid_anchors.any():
            contrastive_loss = torch.tensor(
                0.0, device=sim_c.device, requires_grad=True
            )
        else:
            per_anchor_loss = masked_log_probs.sum(dim=1) / (
                num_pos_per_anchor + self.eps
            )
            contrastive_loss = (
                -per_anchor_loss[valid_anchors].sum() / valid_anchors.sum().float()
            )

        # Compute cross entropy loss
        class_logits_0 = out0[:, : self.d_cat]
        class_logits_1 = out1[:, : self.d_cat]
        class_loss = (
            self.cross_entropy(class_logits_0 / self.logit_temp, labels)
            + self.cross_entropy(class_logits_1 / self.logit_temp, labels)
        ) / 2

        return contrastive_loss + self.class_weight * class_loss

    def eval_forward(self, x):
        """
        Forward pass for evaluation (single view).
        Returns the class logits from the categorical component z_c.
        """
        features = self.head(self.encoder(x))  # (B, d_cat + d_inst)
        z_c = features[:, : self.d_cat]  # Use only the class-specific part
        return z_c
