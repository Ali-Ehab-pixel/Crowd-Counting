from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        
        # 1. Orientation: natural left-right variation
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 2. Geometric: subtle rotation, shift, and shear without losing subjects
        transforms.RandomAffine(
            degrees=10,
            translate=(0.02, 0.02),
            shear=4
        ),
        
        # 3. Photometric: changes in lighting, exposure, and color temperature
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        
        # 4. Sensor & lens simulation: random blur and sharpness variations
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5))
        ], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        
        # 5. Tensor conversion and ImageNet normalization
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])