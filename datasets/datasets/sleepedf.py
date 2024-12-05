import os
import sqlite3
import torch
from operator import itemgetter
from torchvision import transforms
import numpy as np
from easydict import EasyDict


class SleepEDF(torch.utils.data.Dataset):
    def __init__(
        self,
        args: EasyDict,
        target_subject: int,
        is_test: bool = False,
        transform: transforms.Compose = None,
    ):
        if args.mode == "cls":
            if is_test:
                con = sqlite3.connect(
                    os.path.join(
                        args.DB_PATH,
                        f"sleep_edf_test.db",
                    )
                )
                cur = con.cursor()

                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub == {args.target_subject}"
                )
            else:
                con = sqlite3.connect(
                    os.path.join(
                        args.DB_PATH,
                        f"sleep_edf_train.db",
                    )
                )
                cur = con.cursor()

                cur.execute(
                    f"SELECT * FROM MetaData WHERE MetaData.Sub != {args.target_subject}"
                )

            self.metadata = cur.fetchall()
        elif args.mode == "ssl":
            con = sqlite3.connect(os.path.join(args.DB_PATH, f"sleep_edf_train.db"))
            cur = con.cursor()
            cur.execute(f"SELECT * FROM MetaData")
            train_data = cur.fetchall()

            self.metadata = train_data

        else:
            raise ValueError("mode should be either 'cls' or 'ssl'")

        print("LOG >>> Successfully connected to the database")

        self.transform = transform

        cur.close()

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

        data = raw["data"]
        if self.transform is not None:
            data = self.transform(data)

        label = np.array(raw["label"].item().split(","), dtype=np.int64)
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
