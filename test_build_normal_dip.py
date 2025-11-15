import argparse
import os

import numpy as np
import torch
import torch.optim
from absl import app, flags
from skimage.measure import compare_psnr
from torch.autograd import Variable
from torchvision import models
from tqdm import tqdm

from .idip_defend.models.skip import skip
from .idip_defend.utils.common_utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "target_img_name", help="The name of the attacked image", required=True
)
flags.DEFINE_string(
    "save_dir",
    default="./script_logpoint",
    help="The directory used to save the output",
)
flags.DEFINE_integer("SES_lambda", default=0.002, help="Hyperparameter of SES")
flags.DEFINE_string(
    "target_img_domain",
    help="The domain of the attacked image",
    required=True,
    flag_values=["cifar", "imagenet"],
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """

    im_as_ten = torch.from_numpy(cv2im).float()
    im_as_ten.unsqueeze_(0)
    im_as_var = Variable(im_as_ten, requires_grad=False).to(0)
    return im_as_var


def adapative_psnr(img1, img2, size=32):
    psnr, area_cnt = [], 0
    _, h, w = img1.shape

    for i in range(int(h // size)):
        for j in range(int(w // size)):
            img1_part = img1[:, i * size : (i + 1) * size, j * size : (j + 1) * size]
            img2_part = img2[:, i * size : (i + 1) * size, j * size : (j + 1) * size]
            psnr.append(compare_psnr(img1_part, img2_part))
            area_cnt += 1
    psnr = np.array(psnr).min()
    return psnr


def main(argv):
    LAMBDA = FLAGS.SES_lambda
    img_noisy_pil = crop_image(get_image(FLAGS.target_img_name, -1)[0], d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)
    reg_noise_std = 1.0 / 30.0

    if FLAGS.target_img_domain == "cifar":
        input_depth = 1
        num_iter = 1200
        dip_net = skip(
            num_input_channels=input_depth,
            num_output_channels=3,
            num_channels_down=[16, 32, 64, 128],
            num_channels_up=[16, 32, 64, 128],
            num_channels_skip=[4, 4, 4, 4],
            upsample_mode="nearest",
            need_sigmoid=False,
            pad="reflection",
            act_fun="LeakyReLU",
        )
    elif FLAGS.target_img_domain == "imagenet":
        input_depth = 3
        num_iter = 2400
        dip_net = skip(
            num_input_channels=input_depth,
            num_output_channels=3,
            num_channels_down=[16, 32, 64, 128, 128],
            num_channels_up=[16, 32, 64, 128, 128],
            num_channels_skip=[4, 4, 4, 4, 4],
            upsample_mode="nearest",
            need_sigmoid=False,  # remove
            pad="reflection",
            act_fun="LeakyReLU",
        )
    else:
        raise ValueError(f"Unsupported target image domain: {FLAGS.target_img_domain}")

    net = dip_net.type(dtype)
    series, out_series, delta_series = [], [], []
    psnr_max_img = None
    SES_img = None
    net_input = (
        get_noise(input_depth, "noise", (img_noisy_pil.size[1], img_noisy_pil.size[0]))
        .type(dtype)
        .detach()
    )

    # Compute number of parameters
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print("Number of params: %d" % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    def closure():

        global net_input
        global psnr_max_img, SES_img

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        total_loss = mse(out.mean(dim=0, keepdim=True), img_noisy_torch)
        total_loss.backward()

        psrn_gt = adapative_psnr(img_noisy_np, out.detach().cpu().numpy().mean(axis=0))

        if len(series) == 0:
            series.append(psrn_gt)
            out_series.append(psrn_gt)
        elif len(series) == 1:
            series.append(psrn_gt)
            delta_series.append(series[1] - series[0])
            out_series.append(
                LAMBDA * series[-1] + (1 - LAMBDA) * (out_series[-1] + delta_series[-1])
            )
        else:
            series.append(psrn_gt)
            s = LAMBDA * series[-1] + (1 - LAMBDA) * (out_series[-1] + delta_series[-1])
            t = LAMBDA * (s - out_series[-1]) + (1 - LAMBDA) * (delta_series[-1])
            out_series.append(s)
            delta_series.append(t)
            if out_series[-1] > np.array(out_series[:-1]).max():
                SES_img = out.detach().cpu().numpy().mean(axis=0)

        return total_loss

    p = get_params("net,input", net, net_input)

    optimizer = torch.optim.Adam(p, lr=0.01)
    for j in tqdm(range(num_iter), desc="DIP Denoising iterations"):
        optimizer.zero_grad()
        closure()
        optimizer.step()

    np.save(
        os.path.join(
            FLAGS.save_dir, f"SES_defended_{os.path.basename(FLAGS.target_img_name)}"
        ),
        SES_img,
    )


if __name__ == "__main__":
    app.run(main)
