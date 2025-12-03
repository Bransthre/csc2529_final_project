import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from idip_defend.models.skip import skip


def get_params(opt_over, net, net_input, downsampler=None):
    """Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    """
    opt_over_list = opt_over.split(",")
    params = []

    for opt in opt_over_list:

        if opt == "net":
            params += [x for x in net.parameters()]
        elif opt == "down":
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == "input":
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, "what is it?"

    return params


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]


def adapative_psnr(img1, img2, size=32):
    """
    Calculates the minimum PSNR over all size x size patches.

    Args:
        img1: torch.Tensor of shape C x H x W
        img2: torch.Tensor of shape C x H x W
        size: int, size of the patches
    Returns:
        psnr: float, minimum PSNR over all patches
    """
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


def inform_about_attack(original_logits, adversarial_logits, y_example):
    """
    Prints the original and adversarial predictions of the model.

    Args:
        original_logits: torch.Tensor, logits of the original image
        adversarial_logits: torch.Tensor, logits of the adversarial image
        y_example: torch.Tensor, correct label of the image
    """
    print("Current image correct y prediction:", y_example.item())
    print(
        "Current image original prediction:",
        torch.argmax(original_logits, dim=1).item(),
    )
    print(
        "Current image adversarial prediction:",
        torch.argmax(adversarial_logits, dim=1).item(),
    )


def get_net_from_domain(image_domain):
    """
    Returns DIP network and its hyperparameters based on the image domain.

    Args:
        image_domain: str, either "cifar" or "imagenet"
    Returns:
        dip_net: torch.nn.Module, DIP network
        hyperparams: dict, hyperparameters of the DIP network
    """
    if image_domain == "cifar":
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
    elif image_domain == "imagenet":
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
    return dip_net, {"input_depth": input_depth, "num_iter": num_iter}


def spectral_anchoring_loss(mask_alpha, img_hw, device, anchor_to="defense"):
    """
    Returns a spectral anchoring loss function.

    Args:
        mask_alpha: float, parameter for the Gaussian mask in the frequency domain
        img_hw: tuple, (height, width) of the images
        device: torch.device, device to run the computations on
        anchor_to: str, either "defense" or "attack", indicating which image to anchor to
    Returns:
        spectral_anchoring_loss_fn: function, spectral anchoring loss function
    """
    H, W = img_hw
    fy = torch.fft.fftfreq(H, device=device).reshape(-1, 1)
    fx = torch.fft.fftfreq(W, device=device).reshape(1, -1)
    fr = torch.sqrt(fx**2 + fy**2)
    fourier_mask = torch.exp(-mask_alpha * (fr**2)).unsqueeze(0)

    def _spectral_anchoring_loss_def(attack_img, defense_img, net_output):
        target = defense_img
        fourier_target = torch.fft.fft2(target).squeeze()
        fourier_output = torch.fft.fft2(net_output).squeeze()
        fourier_weighted_diff = fourier_output - fourier_target
        spectral_loss = (torch.abs(fourier_weighted_diff).pow(2) * fourier_mask).mean()
        return spectral_loss  # scale down

    def _spectral_anchoring_loss_att(attack_img, defense_img, net_output):
        target = attack_img
        fourier_target = torch.fft.fft2(target).squeeze()
        fourier_output = torch.fft.fft2(net_output).squeeze()
        fourier_weighted_diff = fourier_output - fourier_target
        spectral_loss = (torch.abs(fourier_weighted_diff).pow(2) * fourier_mask).mean()
        return spectral_loss  # scale down

    if anchor_to == "defense":
        return _spectral_anchoring_loss_def
    else:
        return _spectral_anchoring_loss_att


def fake_loss(device):
    """
    Returns a fake loss function that always returns zero.

    Args:
        device: torch.device, device to run the computations on
    Returns:
        fake_loss_fn: function, fake loss function
    """

    def _fake_loss(*args, **kwargs):
        return torch.tensor(0.0, device=device)

    return _fake_loss
