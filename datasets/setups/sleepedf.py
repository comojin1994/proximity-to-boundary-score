import argparse
import glob
import math
import ntpath
import os
import shutil
from datetime import datetime
import numpy as np
from mne.io import read_raw_edf
import dhedfreader
import sqlite3
import traceback
from tqdm import tqdm
from glob import glob


# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {"W": W, "N1": N1, "N2": N2, "N3": N3, "REM": REM, "UNKNOWN": UNKNOWN}

class_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM", 5: "UNKNOWN"}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
}

EPOCH_SEC_SIZE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/opt/workspace/Datasets/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette",
        help="File path to the CSV or NPY file that contains walking data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/workspace/Datasets/physionet_sleep",
        help="Directory where to save outputs.",
    )
    parser.add_argument(
        "--select_ch",
        type=str,
        default="EEG Fpz-Cz",
        help="File path to the trained model used to estimate walking speeds.",
    )
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):
        # if not "ST7171J0-PSG.edf" in psg_fnames[i]:
        #     continue

        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        sampling_rate = raw.info["sfreq"]
        raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
        raw_ch_df = raw_ch_df.to_frame()
        raw_ch_df.set_index(np.arange(len(raw_ch_df)))

        # Get raw header
        f = open(psg_fnames[i], "r", encoding="iso-8859-1")
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()
        raw_start_dt = datetime.strptime(h_raw["date_time"], "%Y-%m-%d %H:%M:%S")

        # Read annotation and its header
        f = open(ann_fnames[i], "r", encoding="iso-8859-1")
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = list(zip(*reader_ann.records()))
        f.close()
        ann_start_dt = datetime.strptime(h_ann["date_time"], "%Y-%m-%d %H:%M:%S")

        # Assert that raw and annotation files start at the same time
        assert raw_start_dt == ann_start_dt

        # Generate label and remove indices
        remove_idx = []  # indicies of the data that will be removed
        labels = []  # indicies of the data that have labels
        label_idx = []
        for a in ann[0]:
            onset_sec, duration_sec, ann_char = a
            ann_str = "".join(ann_char)
            label = ann2label[ann_str]
            if label != UNKNOWN:
                if duration_sec % EPOCH_SEC_SIZE != 0:
                    raise Exception("Something wrong")
                duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                label_epoch = np.ones(duration_epoch, dtype=np.int) * label
                labels.append(label_epoch)
                idx = int(onset_sec * sampling_rate) + np.arange(
                    duration_sec * sampling_rate, dtype=np.int
                )
                label_idx.append(idx)

                print(
                    "Include onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str
                    )
                )
            else:
                idx = int(onset_sec * sampling_rate) + np.arange(
                    duration_sec * sampling_rate, dtype=np.int
                )
                remove_idx.append(idx)

                print(
                    "Remove onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str
                    )
                )
        labels = np.hstack(labels)

        print("before remove unwanted: {}".format(np.arange(len(raw_ch_df)).shape))
        if len(remove_idx) > 0:
            remove_idx = np.hstack(remove_idx)
            select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
        else:
            select_idx = np.arange(len(raw_ch_df))
        print("after remove unwanted: {}".format(select_idx.shape))

        # Select only the data with labels
        print("before intersect label: {}".format(select_idx.shape))
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
        print("after intersect label: {}".format(select_idx.shape))

        # Remove extra index
        if len(label_idx) > len(select_idx):
            print(
                "before remove extra labels: {}, {}".format(
                    select_idx.shape, labels.shape
                )
            )
            extra_idx = np.setdiff1d(label_idx, select_idx)
            # Trim the tail
            if np.all(extra_idx > select_idx[-1]):
                n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                n_label_trims = int(
                    math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate))
                )
                select_idx = select_idx[:-n_trims]
                labels = labels[:-n_label_trims]
            print(
                "after remove extra labels: {}, {}".format(
                    select_idx.shape, labels.shape
                )
            )

        # Remove movement and unknown stages if any
        raw_ch = raw_ch_df.values[select_idx]

        # Verify that we can split into 30-s epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

        # Get epochs and their corresponding labels
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        # Select on sleep periods
        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(y):
            end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print(("Data before selection: {}, {}".format(x.shape, y.shape)))
        x = x[select_idx]
        y = y[select_idx]
        print(("Data after selection: {}, {}".format(x.shape, y.shape)))

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        print("\n=======================================\n")


def make_db(task: str):
    # Define DB
    BASE_PATH = "/opt/workspace/Datasets/physionet_sleep"
    DB_PATH = "/opt/workspace/proximity-to-boundary-score/databases"
    file_idx = 0
    EPOCH_LENGTH = 30
    SAVE_PATH = "/opt/workspace/Datasets/physionet_sleep_preprocessed_" + str(
        EPOCH_LENGTH
    )
    OVERLAP_IVAL = EPOCH_LENGTH // 4

    # Make DB
    con = sqlite3.connect(os.path.join(DB_PATH, f"sleep_edf_{task}_{EPOCH_LENGTH}.db"))
    print("LOG >>> Successfully connected to the training database")

    cur = con.cursor()
    print("LOG >>> Successfully created Table")

    cur.execute(
        """CREATE TABLE MetaData(
        Sub Integer,
        Label text,
        Path text);"""
    )

    filelist = sorted(glob(BASE_PATH + "/*"))

    pbar = tqdm(filelist)
    for filename in filelist:
        try:
            pbar.set_postfix({"Filename": filename})
            raw = np.load(filename)

            x, y = raw["x"], raw["y"]

            overlap_ival = OVERLAP_IVAL if task == "train" else EPOCH_LENGTH

            for jdx in range(0, x.shape[0], overlap_ival):
                if jdx + EPOCH_LENGTH >= x.shape[0]:
                    current_data = x[jdx : jdx + EPOCH_LENGTH, ...]
                    insufficient_num = EPOCH_LENGTH - current_data.shape[0]
                    padding_data = np.repeat(
                        current_data[-1:], repeats=insufficient_num, axis=0
                    )
                    current_data = np.concatenate((current_data, padding_data), axis=0)

                    current_label = y[jdx : jdx + EPOCH_LENGTH]
                    padding_label = np.repeat(9, repeats=insufficient_num, axis=0)
                    current_label = np.concatenate(
                        (current_label, padding_label), axis=0
                    )
                else:
                    current_data = x[jdx : jdx + EPOCH_LENGTH, ...]
                    current_label = y[jdx : jdx + EPOCH_LENGTH]

                current_data = current_data.transpose(0, 2, 1)
                current_label = list(map(str, current_label.tolist()))
                current_label = ",".join(current_label)

                save_filename = os.path.join(SAVE_PATH, task, f"{file_idx:06d}.npz")
                np.savez(save_filename, data=current_data, label=current_label)

                cur.execute(
                    "INSERT INTO MetaData Values(:Sub, :Label, :Path)",
                    {
                        "Sub": int(filename.split("/")[-1][3:5]),
                        "Label": current_label,
                        "Path": save_filename,
                    },
                )

                file_idx += 1
        except Exception as e:
            print(filename)
            print(e)
            print(traceback.format_exc())

    con.commit()
    con.close()


if __name__ == "__main__":
    # main()
    make_db(task="train")
    make_db(task="test")
