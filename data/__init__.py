from .dataset import InpaintingData, MoNuSegTestDataset
from torch.utils.data import DataLoader


def sample_data(loader): 
    while True:
        for batch in loader:
            yield batch


def create_loader(args): 
    dataset = InpaintingData(args)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    
    return data_loader, sample_data(data_loader)


def create_test_loader(args):
    dataset = MoNuSegTestDataset(args.test_path, args.test_img_size)
    return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)