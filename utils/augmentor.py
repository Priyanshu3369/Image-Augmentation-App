import os
import cv2
import uuid
import numpy as np
import albumentations as A
import random
import matplotlib.pyplot as plt

def save_augmented(image_array, output_folder, suffix):
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    filename = f"{uuid.uuid4().hex}_{suffix}.jpg"
    output_path = os.path.join(output_folder, filename)

    # Convert RGB to BGR for OpenCV saving
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)

    return output_path

def process_image(filepath, mode, manual_ops, output_folder, count=50):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_images = []

    if mode == 'auto':
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
            A.RandomCrop(
                height=image.shape[0] - min(10, image.shape[0] // 10),
                width=image.shape[1] - min(10, image.shape[1] // 10),
                p=0.5
            )
        ])
        for i in range(count):
            augmented = transforms(image=image)['image']
            augmented = np.clip(augmented, 0, 255).astype(np.uint8)

            # Debug preview (optional)
            if i == 0:
                plt.imshow(augmented)
                plt.title("Preview: Auto-Augmented")
                plt.axis('off')
                plt.show()

            path = save_augmented(augmented, output_folder, f"auto_{i}")
            output_images.append(path)

    elif mode == 'manual':
        per_aug_count = count // len(manual_ops) if manual_ops else 0

        for aug in manual_ops:
            for i in range(per_aug_count):
                if aug == 'flip':
                    aug_img = cv2.flip(image, 1)

                elif aug == 'rotate':
                    rotate_aug = A.Rotate(limit=random.randint(15, 45), p=1.0)
                    aug_img = rotate_aug(image=image)['image']

                elif aug == 'brightness':
                    bright_aug = A.RandomBrightnessContrast(p=1.0)
                    aug_img = bright_aug(image=image)['image']

                elif aug == 'noise':
                    noise_aug = A.GaussNoise(var_limit=(5.0, 20.0), p=1.0)
                    aug_img = noise_aug(image=image)['image']

                elif aug == 'crop':
                    crop_aug = A.RandomCrop(
                        height=image.shape[0] - min(10, image.shape[0] // 10),
                        width=image.shape[1] - min(10, image.shape[1] // 10),
                        p=1.0
                    )
                    aug_img = crop_aug(image=image)['image']

                else:
                    continue

                aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

                # Optional preview for 1st manual aug
                if i == 0:
                    plt.imshow(aug_img)
                    plt.title(f"Preview: {aug} Manual")
                    plt.axis('off')
                    plt.show()

                path = save_augmented(aug_img, output_folder, f"{aug}_{i}")
                output_images.append(path)

    return output_images
