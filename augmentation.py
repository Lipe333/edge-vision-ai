import os
import cv2
from tqdm import tqdm
import albumentations as A

INPUT_DIR = "croppedImages/val"
OUTPUT_DIR = "croppedImages_aug/val"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# pipeline de augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
])

AUG_PER_IMAGE = 3

classes = os.listdir(INPUT_DIR)

for cls in classes:

    input_class_dir = os.path.join(INPUT_DIR, cls)
    output_class_dir = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(output_class_dir, exist_ok=True)

    images = os.listdir(input_class_dir)

    for img_name in tqdm(images, desc=f"Class {cls}"):

        img_path = os.path.join(input_class_dir, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        # salva original
        cv2.imwrite(
            os.path.join(output_class_dir, img_name),
            image
        )

        # gera augmentations
        for i in range(AUG_PER_IMAGE):

            augmented = transform(image=image)
            aug_img = augmented["image"]

            new_name = img_name.replace(".jpg", f"_aug{i}.jpg")

            cv2.imwrite(
                os.path.join(output_class_dir, new_name),
                aug_img
            )