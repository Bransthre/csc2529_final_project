import copy
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchattacks
from absl import app, flags
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from idip_defend.models.ewc import EWC
from idip_defend.utils.common_utils import (
    adapative_psnr,
    fake_loss,
    get_net_from_domain,
    get_noise,
    get_params,
    inform_about_attack,
    np_to_torch,
    spectral_anchoring_loss,
)
from idip_defend.utils.imagenet.dataset_utils import (
    get_imagenet_dataset,
    transform_layer,
)

warnings.simplefilter("ignore")

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
flags.DEFINE_boolean(
    "update_ewc_after_defense",
    default=False,
    help="Whether to update EWC after each defense",
)
flags.DEFINE_float("SES_lambda", default=0.002, help="Hyperparameter of SES")
flags.DEFINE_integer("attack_length", default=5, help="Length of attack sequence")
flags.DEFINE_float("past_task_weight", default=1.0, help="Weight for past tasks in EWC")
flags.DEFINE_string("anchoring_loss_fn", default="mse", help="Anchoring loss function")
flags.DEFINE_float("fourier_mask_alpha", default=10.0, help="Alpha for fourier mask")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
mse_loss = torch.nn.MSELoss().type(dtype)


def dip_output_of_image(
    per_image_dip_net: nn.Sequential,
    attacked_img,
    input_depth,
    reg_noise_std=1.0 / 30.0,
):
    net_input = (
        get_noise(input_depth, "noise", attacked_img.shape[1:]).type(dtype).detach()
    )

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = per_image_dip_net(net_input)

    return out.mean(axis=0, keepdims=True)


# Per-image routine
def train_dip_on_image(
    per_image_dip_net: nn.Sequential,
    attacked_img,
    num_iter,
    input_depth,
    ses_parameter,
    dip_objective: EWC,
    anchoring_loss_fn,
    reg_noise_std=1.0 / 30.0,
    past_examples=[],
):
    series, out_series, delta_series = [], [], []
    psnr_max_img, SES_img = None, None
    net_input = (
        get_noise(input_depth, "noise", attacked_img.shape[1:]).type(dtype).detach()
    )

    # Loss
    img_noisy_torch = np_to_torch(attacked_img).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    past_example_losses = [[] for _ in past_examples]

    def get_past_example_loss(past_example, anchoring_loss_fn):
        past_example_out = dip_output_of_image(
            per_image_dip_net, past_example, input_depth, reg_noise_std
        )
        past_example_noisy_torch = np_to_torch(past_example).type(dtype)
        past_example_loss = anchoring_loss_fn(
            past_example_noisy_torch,
            past_example_out.mean(dim=0, keepdim=True),
        )
        return past_example_loss, past_example_out

    # Can probably clean the series thing up a bit more
    def closure(net_input, psnr_max_img, SES_img):
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = per_image_dip_net(net_input)

        total_loss = dip_objective(
            img_noisy_torch,
            out.mean(dim=0, keepdim=True),
            list(per_image_dip_net.parameters()),
        )

        past_example_outs = []

        for idx, past_example in enumerate(past_examples):
            past_example_out, past_example_anchoring_loss = get_past_example_loss(
                past_example, anchoring_loss_fn
            )
            past_example_outs.append(past_example_out)
            past_example_losses[idx].append(past_example_anchoring_loss.item())
            total_loss += past_example_anchoring_loss

        total_loss.backward()
        psrn_gt = adapative_psnr(attacked_img, out.detach().cpu().numpy().mean(axis=0))

        if len(series) == 0:
            series.append(psrn_gt)
            out_series.append(psrn_gt)
        elif len(series) == 1:
            series.append(psrn_gt)
            delta_series.append(series[1] - series[0])
            out_series.append(
                ses_parameter * series[-1]
                + (1 - ses_parameter) * (out_series[-1] + delta_series[-1])
            )
        else:
            series.append(psrn_gt)
            s = ses_parameter * series[-1] + (1 - ses_parameter) * (
                out_series[-1] + delta_series[-1]
            )
            t = ses_parameter * (s - out_series[-1]) + (1 - ses_parameter) * (
                delta_series[-1]
            )
            out_series.append(s)
            delta_series.append(t)
            if out_series[-1] > np.array(out_series[:-1]).max():
                SES_img = out.detach().cpu().numpy().mean(axis=0)
            if series[-1] > np.array(series[:-1]).max():
                psnr_max_img = out.detach().cpu().numpy().mean(axis=0)

        return total_loss, (net_input, psnr_max_img, SES_img), past_example_outs

    total_params = get_params("net,input", per_image_dip_net, net_input)

    optimizer = torch.optim.Adam(total_params, lr=0.01)
    dip_losses = []
    for n_train_iter in tqdm(range(num_iter), desc="DIP Denoising iterations"):
        optimizer.zero_grad()
        dip_loss, (net_input, psnr_max_img, SES_img), past_examples_curr_outputs = (
            closure(net_input, psnr_max_img, SES_img)
        )
        dip_losses.append(dip_loss)
        optimizer.step()

    if SES_img is None:
        SES_img = psnr_max_img

    return (
        psnr_max_img,
        SES_img,
        {
            "series": series[::100],
            "out_series": out_series[::100],
            "delta_series": delta_series[::100],
            "dip_losses": dip_losses[::100],
            "past_example_losses": [arr[::100] for arr in past_example_losses],
            "past_examples_curr_outputs": past_examples_curr_outputs,
        },
    )


def main(argv):
    print("Use the PGD Attack by default for testing.")
    print("Torch", torch.__version__, "CUDA", torch.version.cuda)
    print("Device:", torch.device("cuda:0"))

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"Using device '{dev}'")

    # Image loading
    resnet50 = models.resnet50(pretrained=True).to(dev)
    resnet50.eval()
    resnet_classifier = copy.deepcopy(resnet50)
    resnet_classifier = nn.Sequential(transform_layer(), resnet_classifier)
    resnet_classifier.eval()

    dip_net, net_info = get_net_from_domain(FLAGS.target_img_domain)
    input_depth = net_info["input_depth"]
    num_iter = net_info["num_iter"]
    dip_net = dip_net.type(dtype)

    validation_set = get_imagenet_dataset("val")
    example_dataloader = DataLoader(
        validation_set, batch_size=1, shuffle=True, drop_last=True
    )
    tk = torchattacks.PGD(resnet_classifier, eps=0.03, alpha=0.005, steps=40)

    if FLAGS.anchoring_loss_fn == "mse":
        anchoring_loss_fn = mse_loss
    elif FLAGS.anchoring_loss_fn == "spectral":
        anchoring_loss_fn = spectral_anchoring_loss(
            FLAGS.fourier_mask_alpha, (224, 224), dev
        )
    elif FLAGS.anchoring_loss_fn == "fake":
        anchoring_loss_fn = fake_loss(dev)
    else:
        raise ValueError(
            f"Unsupported anchoring loss function: {FLAGS.anchoring_loss_fn}"
        )

    num_attacks_so_far = 0
    ses_parameter = FLAGS.SES_lambda
    prev_x_examples = []
    dip_objective = EWC(
        main_task_objective=mse_loss,
        past_task_weight=FLAGS.past_task_weight,
    ).to(dev)

    for x_example, y_example in example_dataloader:
        x_example = x_example.to(dev)
        y_example = y_example.to(dev)
        true_y_repr = torch.zeros(1, 1000).to(dev)
        true_y_repr[0, y_example] = 1.0
        adv_image = tk(x_example, true_y_repr)
        original_logits = resnet_classifier(x_example)
        adversarial_logits = resnet_classifier(adv_image)
        inform_about_attack(original_logits, adversarial_logits, y_example)

        adv_image_input = adv_image.detach().cpu().numpy()[0]

        dip_defend_output = train_dip_on_image(
            per_image_dip_net=dip_net,  # This net will be updated constantly
            attacked_img=adv_image_input,
            num_iter=num_iter,
            input_depth=input_depth,
            ses_parameter=ses_parameter,
            dip_objective=dip_objective,
            anchoring_loss_fn=anchoring_loss_fn,
            past_examples=prev_x_examples,
        )
        psnr_max_img, ses_img, defense_info = dip_defend_output
        dip_logits_psnrmax = resnet_classifier(
            torch.from_numpy(psnr_max_img).unsqueeze(0).type(dtype).to(dev)
        )
        dip_logits_ses = resnet_classifier(
            torch.from_numpy(ses_img).unsqueeze(0).type(dtype).to(dev)
        )

        # Evaluate logits for past examples
        print("Evaluating past examples...")
        past_examples_logits = []
        with torch.no_grad():
            for idx, past_example in enumerate(prev_x_examples):
                past_examples_defended = dip_output_of_image(
                    dip_net, past_example, input_depth
                )
                past_example_defended_logits = resnet_classifier(past_examples_defended)
                past_examples_logits.append(
                    past_example_defended_logits.detach().cpu().numpy()
                )

        print("Saving defense results...")
        save_destination = os.path.join(
            FLAGS.save_dir, f"attack_{num_attacks_so_far}_status.pt"
        )
        with open(save_destination, "wb") as f:
            torch.save(
                {
                    "original_img": (
                        x_example.detach().cpu().numpy(),
                        original_logits.detach().cpu().numpy(),
                    ),
                    "attacked_img": (
                        adv_image.detach().cpu().numpy(),
                        adversarial_logits.detach().cpu().numpy(),
                    ),
                    "psnr_max_defense_img": (
                        psnr_max_img,
                        dip_logits_psnrmax.detach().cpu().numpy(),
                    ),
                    "ses_defense_img": (ses_img, dip_logits_ses.detach().cpu().numpy()),
                    "defense_info": defense_info,
                    "past_examples_logits": past_examples_logits,
                },
                f,
            )

        num_attacks_so_far += 1
        prev_x_examples.append(adv_image_input)
        if FLAGS.update_ewc_after_defense:
            net_input = (
                get_noise(input_depth, "noise", adv_image_input.shape[1:])
                .type(dtype)
                .detach()
            )
            img_noisy_torch = np_to_torch(adv_image_input).type(dtype)
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            net_input = net_input_saved + (noise.normal_() * 1 / 30.0)
            out = dip_net(net_input)

            final_dip_loss = (
                (out.mean(dim=0, keepdim=True) - img_noisy_torch).pow(2).mean()
            )

            dip_objective.expand_examples(list(dip_net.parameters()), final_dip_loss)

        if num_attacks_so_far >= FLAGS.attack_length:
            break


if __name__ == "__main__":
    app.run(main)
