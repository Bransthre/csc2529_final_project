import copy
import os

import numpy as np
import torch
import torch.optim
import torchattacks
from absl import app, flags
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from idip_defend.models.skip import skip
from idip_defend.utils.common_utils import *
from idip_defend.utils.imagenet.dataset_utils import (
    get_imagenet_dataset,
    transform_layer,
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "target_img_domain",
    default=None,
    help="The domain of the attacked image",
    required=True,
)
flags.DEFINE_string(
    "save_dir",
    default="./script_logpoint",
    help="The directory used to save the output",
)
flags.DEFINE_float("SES_lambda", default=0.002, help="Hyperparameter of SES")
flags.DEFINE_integer("attack_length", default=5, help="Length of attack sequence")

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

    print("Use the PGD Attack by default for testing.")
    print("Torch", torch.__version__, "CUDA", torch.version.cuda)
    print("Device:", torch.device("cuda:0"))
    dev = None

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"Using device '{dev}'")

    if FLAGS.target_img_domain == "cifar":
        raise Exception("Not supporting CIFAR yet.")
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

    dip_net = dip_net.type(dtype)

    # Image loading
    resnet50 = models.resnet50(pretrained=True).to(dev)
    resnet50.eval()
    model = copy.deepcopy(resnet50)
    model = nn.Sequential(transform_layer(), model)
    mse = torch.nn.MSELoss().type(dtype)
    LAMBDA = FLAGS.SES_lambda

    # Per-image routine
    def dip_defend_per_image(
        per_image_dip_net,
        attacked_img,
    ):
        series, out_series, delta_series = [], [], []
        psnr_max_img, SES_img = None, None

        reg_noise_std = 1.0 / 30.0
        net_input = (
            get_noise(input_depth, "noise", attacked_img.shape[1:]).type(dtype).detach()
        )

        # Loss
        img_noisy_torch = np_to_torch(attacked_img).type(dtype)
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        # Can probably clean the series thing up a bit more
        def closure(net_input, psnr_max_img, SES_img):
            if reg_noise_std > 0:
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)

            out = per_image_dip_net(net_input)

            total_loss = mse(out.mean(dim=0, keepdim=True), img_noisy_torch)
            total_loss.backward()

            psrn_gt = adapative_psnr(
                attacked_img, out.detach().cpu().numpy().mean(axis=0)
            )

            if len(series) == 0:
                series.append(psrn_gt)
                out_series.append(psrn_gt)
            elif len(series) == 1:
                series.append(psrn_gt)
                delta_series.append(series[1] - series[0])
                out_series.append(
                    LAMBDA * series[-1]
                    + (1 - LAMBDA) * (out_series[-1] + delta_series[-1])
                )
            else:
                series.append(psrn_gt)
                s = LAMBDA * series[-1] + (1 - LAMBDA) * (
                    out_series[-1] + delta_series[-1]
                )
                t = LAMBDA * (s - out_series[-1]) + (1 - LAMBDA) * (delta_series[-1])
                out_series.append(s)
                delta_series.append(t)
                if out_series[-1] > np.array(out_series[:-1]).max():
                    SES_img = out.detach().cpu().numpy().mean(axis=0)
                if series[-1] > np.array(series[:-1]).max():
                    psnr_max_img = out.detach().cpu().numpy().mean(axis=0)

            return total_loss, (net_input, psnr_max_img, SES_img)

        # Compute number of parameters
        s = sum([np.prod(list(p.size())) for p in per_image_dip_net.parameters()])
        print("Number of params: %d" % s)
        total_params = get_params("net,input", per_image_dip_net, net_input)

        optimizer = torch.optim.Adam(total_params, lr=0.01)
        dip_losses = []
        for n_train_iter in tqdm(range(num_iter), desc="DIP Denoising iterations"):
            optimizer.zero_grad()
            dip_loss, (net_input, psnr_max_img, SES_img) = closure(
                net_input, psnr_max_img, SES_img
            )
            dip_losses.append(dip_loss)
            optimizer.step()

        return psnr_max_img, SES_img, dip_losses

    # Example pipeline 2:
    data = get_imagenet_dataset("val")
    print(f"Validation dataset has {len(data)} items")
    data.get_dataframe().head()
    example_dataloader = DataLoader(data, batch_size=1, shuffle=True, drop_last=True)
    tk = torchattacks.PGD(model, eps=0.03, alpha=0.005, steps=40)
    num_attacks_so_far = 0

    for x_example, y_example in example_dataloader:
        x_example = x_example.to(dev)
        y_example = y_example.to(dev)
        true_y_repr = torch.zeros(1, 1000).to(dev)
        true_y_repr[0, y_example] = 1.0
        adv_image = tk(x_example, true_y_repr)

        original_logits = model(x_example)
        adversarial_logits = model(adv_image)

        dip_defend_output = dip_defend_per_image(
            copy.deepcopy(dip_net),
            adv_image.detach().cpu().numpy()[0],
        )
        psnr_max_img, ses_img, dip_losses = dip_defend_output
        dip_logits_psnrmax = model(psnr_max_img)
        dip_logits_ses = model(ses_img)

        np.save(
            os.path.join(FLAGS.save_dir, f"attack_{num_attacks_so_far}_status"),
            {
                "original_img": (x_example, original_logits),
                "attacked_img": (adv_image, adversarial_logits),
                "psnr_max_defense_img": (psnr_max_img, dip_logits_psnrmax),
                "ses_defense_img": (ses_img, dip_logits_ses),
            },
        )
        num_attacks_so_far += 1
        if num_attacks_so_far >= FLAGS.attack_length:
            break


if __name__ == "__main__":
    app.run(main)
