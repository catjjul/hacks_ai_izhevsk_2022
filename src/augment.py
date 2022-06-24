import albumentations as A
from config import IMAGE_SIZE


train_augmentation = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.45),
                A.RandomRotate90(p=0.05),
            ],
            p=0.7
        ),
        A.OneOf(
            [
                A.HueSaturationValue(p=0.5),
                A.RandomGamma(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
            ],
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.2, rotate_limit=90,
            p=0.3
        )
    ], p=1)


valid_augmentation = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
    ])
