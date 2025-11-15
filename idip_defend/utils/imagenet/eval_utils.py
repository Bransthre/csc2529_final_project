import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from torch.nn.functional import softmax

imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(imagenet_labels_url)
class_names = json.loads(response.text)


def class_name_to_id(class_name):
    """
    Maps ImageNet class name to ID number between 0 and 1000 (exclusive)
    E.g. class_name_to_id('Border Collie') -> int
    """
    for i, cn in enumerate(class_names):
        if cn == class_name:
            return i
    return None


def class_id_to_name(class_id):
    """
    Maps ImageNet class ID back to it's canonical class name
    """
    return class_names[class_id]


def top_5_classes(y, class_names=None):
    """
    y : tensor([1, 1000])
        The output of the model, e.g. from running `model(to_tensor(x))`. The values
        are normalized in this function by using softmax to convert them to probabilities.
    """
    y = torch.Tensor.cpu(y)
    p = softmax(y[0, :], dim=0)
    values, indices = p.topk(5)
    return [
        (class_names[index], value)
        for index, value in zip(indices.detach().numpy(), values.detach().numpy())
    ]


def probability_of_class(y, label):
    """
    If y is the output of a resnet model or similar, and label is an imagenet
    label, either the canonical name as a string, or an integer id, the model's
    confidence in classifying the output as `label` is returned.
    """
    y = torch.Tensor.cpu(y)
    if isinstance(label, str):
        label = class_name_to_id(label)
    if torch.is_tensor(label):
        label = int(torch.Tensor.cpu(label).detach().numpy())
    p = softmax(y[0, :], dim=0)
    return p[label].detach().numpy()


def autolabel(rects, ax, labels):
    """
    Auxiliary function to `plot_prediction`.

    Attach a text label above each bar displaying its height.
    See https://matplotlib.org/2.0.2/examples/api/barchart_demo.html
    """
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            1.05 * height,
            label,
            ha="left",
            va="bottom",
            rotation=45,
        )


def plot_prediction(x, y, ax_img=None, ax_bar=None, title=None):
    """
    Visualises the input image along with the top 5 classifications of the model.

    Attributes
    ------------------
    x : PIL image
        The input image
    y : tensor([1, 1000])
        The output of the model, e.g. from running `model(to_tensor(x))`
    """
    y = torch.Tensor.cpu(y)
    fig = None
    if ax_img is None or ax_bar is None:
        fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    if title is not None:
        assert fig is not None
        fig.suptitle(title, fontsize=18)
    top_classes_str = " ".join([f"{n} ({p:.3})" for n, p in top_5_classes(y)])
    title = f"Top classes: {top_classes_str}"
    # ax.set_title(title)
    axs[0].imshow(x)
    axs[0].axis("off")
    axs[0].set_title("Input image")
    ps = [p[1] for p in top_5_classes(y)]
    names = [p[0] for p in top_5_classes(y)]
    rects = axs[1].bar(np.arange(5), ps)
    if max(ps) > 0.9:
        axs[1].set_ylim([0.0, max(ps) + 0.3])
    else:
        axs[1].set_ylim([0.0, max(ps) + 0.1])
    autolabel(rects, axs[1], names)
    axs[1].set_title("Confidence for top 5 classes")
