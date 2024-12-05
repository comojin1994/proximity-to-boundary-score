from datasets.datasets.bcic2a import BCIC2a
from datasets.datasets.bcic2b import BCIC2b
from datasets.datasets.sleepedf import SleepEDF
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional
import datasets.eeg_transforms as e_transforms

dataset_dict = {
    "bcic2a": BCIC2a,
    "bcic2b": BCIC2b,
    "sleepedf": SleepEDF,
}


def load_data(target_subject, args):
    args.target_subject = target_subject

    transform = transforms.Compose([e_transforms.ToTensor()])

    dataset = DatasetMaker(args.dataset)
    train_dataloader, test_dataloader = dataset.load_data(
        args, transform, target_subject
    )

    return train_dataloader, test_dataloader


class DatasetMaker:

    def __init__(self, dataset_name):
        self.dataset = dataset_dict[dataset_name]

    def load_data(
        self,
        args,
        transform: Optional[transforms.Compose] = None,
        target_subject: Optional[int] = None,
    ):

        train_dataset = self.dataset(
            args=args,
            target_subject=target_subject,
            is_test=False,
            transform=transform,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        test_dataset = self.dataset(
            args=args,
            target_subject=target_subject,
            is_test=True,
            transform=transform,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_dataloader, test_dataloader
