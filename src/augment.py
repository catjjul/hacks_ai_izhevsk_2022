import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.config import IMAGE_SIZE

train_augmentation = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.45),
                A.RandomRotate90(p=0.05),
            ],
            p=1
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.5),
                A.RandomGamma(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ],
            p=1
        ),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=90,
            p=1
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], p=1)
