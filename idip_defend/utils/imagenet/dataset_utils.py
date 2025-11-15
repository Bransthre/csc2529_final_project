import copy
import glob
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
from PIL import Image
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from idip_defend.utils.imagenet.eval_utils import *

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")


#####################################
### IMAGENET DATASET LOADING
#####################################
input_transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224)]  # 224 x 224
)


class transform_layer(nn.Module):
    """
    Applies normalization according to https://pytorch.org/vision/0.8/models.html
    """

    def __init__(self):
        super(transform_layer, self).__init__()
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def forward(self, x):
        # apply the input_transform if it has not already been applied
        if not torch.is_tensor(x) and x.size != (224, 224):
            x = input_transform(x)
        # covert to tensor if not already done
        if not torch.is_tensor(x):
            x = self.to_tensor(x)

        x = x.to(dev)
        # then normalize
        x = self.normalize_transform(x)
        # add batch number
        if len(x.size()) == 3:
            return x.unsqueeze(0)
        else:
            return x


class CustomDataset(Dataset):

    def __init__(self, path: Path, split: str, frac: float = None):
        """
        Attributes
        -------------
        path : Path
          A Path object pointing to the root of the data
        split : str
          A string. Must be one of "val", "test" and "train"
        frac : float
          A value between 0.0 and 1.0. If specified, only this fraction of the found
          data will be used to create the dataset, and the dataset is shuffled before
          doing so.
        """
        assert path.exists()

        # save for __getitem__ later
        self.path = path

        # create the csv of all of our files, ignoring already set up eval,val,train split
        csv_files = list(glob.glob(str(path / f"labels/{split}.csv")))
        # csv_files = list(glob.glob(str(path / "labels/*.csv")))
        data_df = pd.concat([pd.read_csv(f) for f in csv_files])

        # Figure out the mappings from the confusing "n01855672"-like labels
        imagenet_label_map_url = "https://raw.githubusercontent.com/mf1024/ImageNet-Datasets-Downloader/master/classes_in_imagenet.csv"
        response = requests.get(imagenet_label_map_url)
        class_names_df = pd.read_csv(
            io.StringIO(response.content.decode("utf-8"))
        ).set_index("synid")

        # add label column to original dataframe, which previously had "n\d+" labels
        data_df["class_label"] = data_df.apply(
            lambda x: class_names_df.loc[x[1]][0], axis=1
        )

        # then also map to IDs
        imagenet_label_map_url = "https://raw.githubusercontent.com/akshaychawla/ImageNet-downloader/master/imagenet_class_index.json"
        response = requests.get(imagenet_label_map_url)
        class_ids = json.loads(response.text)
        # map from label to ID instead
        self.class_id = {}
        for k in class_ids.keys():
            self.class_id[class_ids[k][0]] = int(k)

        # save for later
        if frac is None:
            self.df = data_df
        else:
            # shuffle the dataset of a fraction is used
            self.df = data_df.sample(frac=frac).reset_index(drop=True)

        # transform function
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        # can do len(dataset)
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Any:
        row = self.df.iloc[index]
        # use pillow to load the images, like we did for Blas, then crop and centre
        image = input_transform(Image.open(self.path / f"images/{row['filename']}"))
        # we created meaningful labels in __init__, so use these
        class_label = str(row["label"])
        # convert to tensors
        return self.to_tensor(image), torch.tensor(self.class_id[class_label])

    def get_dataframe(self):
        return self.df


def evaluate_model(
    model,
    data,
    noise=None,
    criterion=nn.CrossEntropyLoss(reduction="sum"),
    batchsize=50,
    stop_after=100000,
    model_name=None,
    target_label=None,
):
    """
    Evaluates the model's performance in terms of loss, top-1 accuracy and top-5 accuracy
    on images from the data, adding the same `noise` to all the images if
    it is specified.

    Attributes:
    -----------------
    model : nn.Module
      The imagenet-trained module. Should be of shape:
      tensor([x, 3, 224, 224]) -> tensor([x, 1000])
    data : Dataset
      The data to evaluate on
    stop_after : int
      Only does this many batches before stopping the evaluation. Used to limit computation time.
    model_name : str
      If specified, the returned dataframe has an index with the model name in it
    target_label : str
      If specified, evaluation is done in terms of how successfully the specified
      noise can perturb the classification to predict `target_label`
    noise : tensor([3, 224, 224])
      Noise to be applied to each of the input images in `data`
    criterion : fun : tensor([x, 1000]) , tensor([x, 1]) -> tensor([1])
      Loss function to base evaluation on
    """
    with torch.no_grad():
        cumulative_loss = torch.zeros([1]).to(dev)
        correct_top1s = 0
        correct_top5s = 0

        # temporaily put model in GPU
        model = model.to(dev)

        if target_label is not None:
            target_label = (
                torch.scalar_tensor(class_name_to_id(target_label))
                .type(torch.LongTensor)
                .to(dev)
            )

        # repeat the noise, once for each image in the batch
        if noise is not None:
            noise = noise.unsqueeze(0).repeat(batchsize, 1, 1, 1)
            noise = noise.to(dev)

        dataloader = DataLoader(
            data, batch_size=batchsize, shuffle=True, drop_last=True
        )
        for i, (imgs, labels) in enumerate(dataloader):
            torch.cuda.empty_cache()
            # move to GPU if we use one
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            if noise is not None:
                imgs = torch.clamp(imgs + noise, 0, 1)

            # make prediction
            out = model(imgs).detach()
            p = softmax(out, dim=1)
            _, predict_labels = p.topk(5)

            # iterate our batch and count correct predictions
            for lab, pred in zip(labels, predict_labels):
                # We are doing a targeted attack, so evaluate based on how successful that is
                if target_label is not None:
                    lab = target_label
                # top-5
                if lab in pred:
                    correct_top5s += 1
                # top-1
                if lab == pred[0]:
                    correct_top1s += 1
            # accumlate loss
            cumulative_loss += criterion(out, labels)

            if i == stop_after - 1:
                break

        # aggregate
        num_datapoints = min(batchsize * stop_after, len(data))
        cumulative_loss /= num_datapoints
        correct_top1s /= num_datapoints
        correct_top5s /= num_datapoints

        # make sure model is removed from GPU memory
        # model = model.cpu()

        # format result in nice dataframe
        row_entries = [cumulative_loss.item(), correct_top1s, correct_top5s]
        column_names = ["Loss", "Top-1 Accuracy", "Top-5 Accuracy"]
        # we are evaluating how often we can get the target label in top 1/5
        if target_label is not None:
            column_names = ["Loss", "Top-1 Success rate", "Top-5 Success rate"]
        if model_name is not None:
            return pd.DataFrame(
                [[model_name] + row_entries], columns=["Model"] + column_names
            ).set_index("Model")
        else:
            return pd.DataFrame([row_entries], columns=column_names)


def get_imagenet_dataset(split, frac=None):
    assert split in ["train", "val", "test"]
    return CustomDataset(Path("./dataset/mini-imagenet/"), split, frac)
