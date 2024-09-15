import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smoothing: float = 1e-6, num_classes: int = 1) -> None:
        super(DiceLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Dice loss.

        Args:
            prediction: Tensor of shape (batch_size, num_classes, height, width)
            truth: Tensor of shape (batch_size, num_classes, height, width)

        Returns:
            Dice loss value.
        """
        total_loss = 0.0

        truth_one_hot = F.one_hot(truth, num_classes=self.num_classes)
        truth_one_hot = truth_one_hot.permute(0, 3, 1, 2).float()

        for i in range(self.num_classes):
            prediction_mask_i = prediction[:, i, :, :]
            truth_mask_i = truth_one_hot[:, i, :, :]

            prediction_mask_i = prediction_mask_i.contiguous().view(-1)
            truth_mask_i = truth_mask_i.contiguous().view(-1)

            intersection = (prediction_mask_i * truth_mask_i).sum()
            dice_coefficient = (2.0 * intersection + self.smoothing) / (
                prediction_mask_i.sum() + truth_mask_i.sum() + self.smoothing
            )

            total_loss += 1 - dice_coefficient

        return total_loss / self.num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean") -> None:
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Focal loss.

        Args:
            prediction: Tensor of shape (batch_size, num_classes, height, width)
            truth: Tensor of shape (batch_size, height, width)

        Returns:
            Focal loss value.
        """
        N, C, H, W = prediction.shape

        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, C)
        truth = truth.view(-1, 1)

        pt = prediction[range(prediction.size(0)), truth.squeeze()]

        focal_factor = (1 - pt) ** self.gamma

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                if not isinstance(self.alpha, torch.Tensor):
                    self.alpha = torch.tensor(
                        self.alpha, dtype=prediction.dtype, device=prediction.device
                    )
                alpha_t = self.alpha[truth.squeeze()]
                loss = -alpha_t * focal_factor * torch.log(pt)
            else:
                loss = -self.alpha * focal_factor * torch.log(pt)
        else:
            loss = -focal_factor * torch.log(pt)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss