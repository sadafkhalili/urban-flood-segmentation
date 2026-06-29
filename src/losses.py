import torch
from torch import nn


def dice_loss(outputs, masks, num_classes, smooth=1e-6):
    probabilities = torch.softmax(outputs, dim=1)

    masks_one_hot = torch.nn.functional.one_hot(
        masks,
        num_classes=num_classes
    )

    masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float()

    intersection = (probabilities * masks_one_hot).sum(dim=(0, 2, 3))
    total = probabilities.sum(dim=(0, 2, 3)) + masks_one_hot.sum(dim=(0, 2, 3))

    dice = (2 * intersection + smooth) / (total + smooth)

    return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, outputs, masks):
        ce = self.cross_entropy(outputs, masks)
        dice = dice_loss(outputs, masks, self.num_classes)

        return self.ce_weight * ce + self.dice_weight * dice


if __name__ == "__main__":
    outputs = torch.randn(2, 3, 256, 256)
    masks = torch.randint(0, 3, (2, 256, 256))

    criterion = CombinedLoss(num_classes=3)

    loss = criterion(outputs, masks)

    print("Combined loss:", round(loss.item(), 4))
    print("Loss functions are ready.")
