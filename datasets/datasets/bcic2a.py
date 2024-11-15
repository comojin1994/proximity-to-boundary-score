from torchvision import transforms
from easydict import EasyDict
from operator import itemgetter
import os
import sqlite3
import torch
import numpy as np


def join_str_to_make_query(train_sub_list):
    sql = [f"MetaData.Sub == {sub}" for sub in train_sub_list]
    return " OR ".join(sql)


class BCIC2a(torch.utils.data.Dataset):
    """
    * 769: Left
    * 770: Right
    * 771: foot
    * 772: tongue
    """

    def __init__(
        self,
        args: EasyDict,
        target_subject: int,
        is_test: bool = False,
        transform: transforms.Compose = None,
    ):
        con = sqlite3.connect(os.path.join(args.DB_PATH, f"{args.dataset}.db"))
        cur = con.cursor()

        if args.mode == "cls":
            if is_test:
                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub == {target_subject}"
                )
            else:
                train_subject_list = [
                    i for i in range(args.num_subjects) if i != target_subject
                ]

                cur.execute(
                    f"SELECT * FROM MetaData WHERE {join_str_to_make_query(train_subject_list)}"
                )

        elif args.mode == "all":
            cur.execute(f"SELECT * FROM MetaData")

        else:
            raise ValueError("mode should be either 'cls' or 'all'")

        self.metadata = cur.fetchall()
        print("LOG >>> Successfully connected to the database")

        cur.close()

        self.transform = transform
        self.is_test = is_test

        if args.weighted_loss and not is_test:
            scores = np.load(
                f"{args.SCORE_PATH}/{args.score}/{args.dataset}/S{target_subject:02d}.npy"
            )
            self.metadata = np.concatenate(
                [self.metadata, scores.reshape(-1, 1)], axis=1
            )

        if args.pruning_ratio is not None and not is_test:
            self._preprocess(target_subject, args)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.metadata[idx][2]
        raw = np.load(filename)

        data = raw["data"][np.newaxis, ...]
        label = np.array(raw["label"], dtype=np.int64)

        if self.transform is not None:
            data = self.transform(data)

        if len(self.metadata[idx]) == 4:
            score = torch.tensor(np.float64(self.metadata[idx][3])).type_as(data)
            return data, label, score
        else:
            return data, label

    def _preprocess(self, target_subject, args):
        scores = np.load(
            f"{args.SCORE_PATH}/{args.score}/{args.dataset}/S{target_subject:02d}.npy"
        )
        threshold = np.quantile(scores, args.pruning_ratio)
        selected_idx = np.where(scores > threshold)[0]
        self.metadata = itemgetter(*selected_idx)(self.metadata)
