from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize(256), # resizing to nearest power of 2 
    transforms.CenterCrop(224), # resizing to 224x224
    transforms.ToTensor(), # from PIL to tensor -> also scales pixel values from 0-255 to 0-1
    transforms.Normalize(mean=[0.485, 0.56, 0.406], std=[0.229, 0.224, 0.225]) # imagenet specific mean and std values -> RGB channels
])

if __name__ == "__main__":
    from PIL import Image
    import os

    flower_path = 'data/flower_photos/daisy'

    # list comprehension to take only the jpg files
    image_files = [f for f in os.listdir(flower_path) if f.endswith('.jpg')]
    # folder path + 1st jpg image in image_files list
    test_img_path = os.path.join(flower_path, image_files[0])

    img = Image.open(test_img_path)
    print(f"original image size : {img.size}")

    transformed = train_transform(img)
    print(f"resized image size : {transformed.shape}")
    print(f"tensor values range : {transformed.min():.3f} to {transformed.max():.3f}")