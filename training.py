""" Import libraries """

from utils.setup_utils import (
    get_configs,
    init_configs,
    init_settings,
)
from datasets.maker import load_data
from models.maker import ModelMaker
from lightning.pytorch import Trainer
import traceback
import torch

""" Config Setting """

args = get_configs()
args = init_configs(args)
init_settings(args)

""" Define main """


def main():
    for target_subject in range(args.num_subjects):
        train_dataloader, _ = load_data(target_subject, args)

        model_maker = ModelMaker(args.model, args.litmodel)
        model = model_maker.load_model(args)

        ### Training ###
        devices = list(map(int, args.GPU_NUM.split(",")))
        trainer = Trainer(
            max_epochs=args.EPOCHS,
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
            deterministic=True,
            benchmark=False,
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
        )

        torch.cuda.empty_cache()

        if args.mode == "all":
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error:\n\n{e}")
        print(f"Traceback:\n\n{traceback.format_exc()}")
