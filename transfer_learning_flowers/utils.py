from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

train_transform = transforms.Compose([
    transforms.Resize(256), # resizing to nearest power of 2 
    transforms.CenterCrop(224), # resizing to 224x224
    transforms.ToTensor(), # from PIL to tensor -> also scales pixel values from 0-255 to 0-1
    transforms.Normalize(mean=[0.485, 0.56, 0.406], std=[0.229, 0.224, 0.225]) # imagenet specific mean and std values -> RGB channels
])

def get_data_loaders(data_dir, batch_size=32):
    # loading the dataset + assigning class labels automatically -> happening inside ImageFolder func
    dataset = datasets.ImageFolder(data_dir, transform=train_transform) # reads all subfolders name -> sorts them alphabetically and then assigns numbers as labels (eg : daisy - 0)

    # splitting the data into 80-20 (train, test)
    train_size = int(0.8*len(dataset))
    test_size = int(0.2*len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # creating dataloaders which batches images for training + testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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
    print("transform working")

    print("testing data loaders")
    train_loader, test_loader = get_data_loaders('data/flower_photos', batch_size=32)

    print(f"train batches : {len(train_loader)}")
    print(f"test batches : {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"batch shape : {images.shape}")
    print(f"labels shape : {labels.shape}")
    print(f"first 5 labels values : {labels[:5]}")
    print("data loaders working")