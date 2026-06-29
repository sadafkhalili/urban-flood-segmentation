import torch


def pixel_accuracy(outputs, masks):
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == masks).sum().item()
    total = masks.numel()
    return correct / total


def mean_iou(outputs, masks, num_classes, smooth=1e-6):
    predictions = torch.argmax(outputs, dim=1)
    iou_scores = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        mask_class = masks == class_id

        intersection = (pred_class & mask_class).sum().float()
        union = (pred_class | mask_class).sum().float()

        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)

    return torch.mean(torch.stack(iou_scores)).item()


def mean_dice(outputs, masks, num_classes, smooth=1e-6):
    predictions = torch.argmax(outputs, dim=1)
    dice_scores = []

    for class_id in range(num_classes):
        pred_class = predictions == class_id
        mask_class = masks == class_id

        intersection = (pred_class & mask_class).sum().float()
        total = pred_class.sum().float() + mask_class.sum().float()

        dice = (2 * intersection + smooth) / (total + smooth)
        dice_scores.append(dice)

    return torch.mean(torch.stack(dice_scores)).item()


if __name__ == "__main__":
    outputs = torch.randn(2, 3, 256, 256)
    masks = torch.randint(0, 3, (2, 256, 256))

    print("Pixel Accuracy:", round(pixel_accuracy(outputs, masks), 4))
    print("Mean IoU:", round(mean_iou(outputs, masks, 3), 4))
    print("Mean Dice:", round(mean_dice(outputs, masks, 3), 4))
