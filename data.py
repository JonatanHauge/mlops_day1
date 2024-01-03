import torch
import numpy

def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    

    train0 = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_images_0.pt")
    train0_targets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_target_0.pt")

    train1 = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_images_1.pt")
    train1_targets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_target_1.pt")

    train2 = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_images_2.pt")
    train2_targets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_target_2.pt")

    train3 = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_images_3.pt")
    train3_targets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_target_3.pt")

    train4 = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_images_4.pt")
    train4_targets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_target_4.pt")

    train5 = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_images_5.pt")
    train5_targets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\train_target_5.pt")

    train = torch.cat((train0, train1, train2, train3, train4, train5), 0)
    train_targets = torch.cat((train0_targets, train1_targets, train2_targets, train3_targets, train4_targets, train5_targets), 0)
    
    test = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\test_images.pt")
    test_tartgets = torch.load(r"C:\Users\johau\Desktop\mlops\dtu_mlops\data\corruptmnist\test_target.pt")

    train = torch.utils.data.TensorDataset(train, train_targets)
    test = torch.utils.data.TensorDataset(test, test_tartgets)
    return train, test 
