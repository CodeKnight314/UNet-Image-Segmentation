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
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class CompositeLoss(nn.Module):
    def __init__(self, dice_weight=0.3, focal_weight=0.3, ce_weight=0.4, 
                 dice_smoothing=1e-6, dice_num_classes=6, focal_gamma=2.0, focal_alpha=None):
        """
        A composite loss function combining Dice Loss, Focal Loss, and Cross Entropy Loss.

        Args:
            dice_weight: Weight for the Dice Loss component.
            focal_weight: Weight for the Focal Loss component.
            ce_weight: Weight for the Cross Entropy Loss component.
            dice_smoothing: Smoothing factor for Dice Loss to avoid division by zero.
            dice_num_classes: Number of classes for Dice Loss (for one-hot encoding).
            focal_gamma: Focusing parameter for Focal Loss (default is 2).
            focal_alpha: Class weights for Focal Loss (to deal with imbalanced datasets).
        """
        super(CompositeLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(smoothing=dice_smoothing, num_classes=dice_num_classes)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
        Calculates the composite loss as a weighted sum of Dice Loss, Focal Loss, and Cross Entropy Loss.

        Args:
            prediction: The model's predicted output of shape (batch_size, num_classes, height, width).
            truth: The ground truth segmentation mask of shape (batch_size, height, width).

        Returns:
            The combined loss value.
        """
        dice_loss_value = self.dice_loss(prediction, truth)
        focal_loss_value = self.focal_loss(prediction, truth)
        ce_loss_value = self.ce_loss(prediction, truth)

        total_loss = self.dice_weight * dice_loss_value + self.focal_weight * focal_loss_value + self.ce_weight * ce_loss_value

        return total_loss

def test_loss_function():
    """
    Test the DiceLoss and FocalLoss functions with sample data.
    """
    num_classes = 6
    batch_size = 16
    height = 256
    width = 256

    prediction_logits = torch.randn(batch_size, num_classes, height, width, requires_grad=True)

    prediction_probs = F.softmax(prediction_logits, dim=1)

    truth = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.long)

    dice_loss_fn = DiceLoss(smoothing=1e-6, num_classes=num_classes)
    focal_loss_fn = FocalLoss(alpha=None, gamma=2.0, reduction='mean')

    dice_loss_value = dice_loss_fn(prediction_probs, truth)

    focal_loss_value = focal_loss_fn(prediction_logits, truth)

    dice_loss_value.backward(retain_graph=True)  
    focal_loss_value.backward()

    print(f"Dice Loss: {dice_loss_value.item()}")
    print(f"Focal Loss: {focal_loss_value.item()}")

if __name__ == "__main__": 
    test_loss_function()