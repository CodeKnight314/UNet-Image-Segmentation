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
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        N, C, H, W = inputs.size()
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.view(-1)

        logpt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)

        logpt = logpt.gather(1, targets.unsqueeze(1))
        pt = pt.gather(1, targets.unsqueeze(1))

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets].unsqueeze(1)
            loss = -at * ((1 - pt) ** self.gamma) * logpt
        else:
            loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class CompositeLoss(nn.Module):
    def __init__(self, dice_weight=0.3, focal_weight=0.3, ce_weight=1.0, 
                 dice_smoothing=1e-6, dice_num_classes=6, focal_gamma=2.0, focal_alpha=None, class_weights=None):
        super(CompositeLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(smoothing=dice_smoothing, num_classes=dice_num_classes)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights, reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        probabilities = F.softmax(prediction, dim=1)
        dice_loss_value = self.dice_loss(probabilities, truth)
        
        focal_loss_value = self.focal_loss(prediction, truth)
        ce_loss_value = self.ce_loss(prediction, truth)

        total_loss = (
            self.dice_weight * dice_loss_value +
            self.focal_weight * focal_loss_value +
            self.ce_weight * ce_loss_value
        )

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