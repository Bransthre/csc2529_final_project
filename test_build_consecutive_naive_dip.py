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
flags.DEFINE_integer(
    "update_ewc",
    default=0,
    help="Whether to update EWC after each defense",
)
flags.DEFINE_float("SES_lambda", default=0.002, help="Hyperparameter of SES")
flags.DEFINE_integer("attack_length", default=5, help="Length of attack sequence")
flags.DEFINE_float("past_task_weight", default=1.0, help="Weight for past tasks in EWC")
flags.DEFINE_string("anchoring_loss_fn", default="mse", help="Anchoring loss function")
flags.DEFINE_string(
    "anchor_to",
    default="defense",
    help="Whether to anchor to 'defense' or 'attack' images in spectral loss",
)
flags.DEFINE_float("fourier_mask_alpha", default=10.0, help="Alpha for fourier mask")
flags.DEFINE_float(
    "anchoring_loss_weight", default=1.0, help="Weight for anchoring loss"
)
flags.DEFINE_integer(
    "num_iter_per_image", default=2400, help="Number of DIP iterations"
)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
mse_loss = torch.nn.MSELoss().type(dtype)


def dip_output_of_image(
    per_image_dip_net: nn.Sequential,
    net_input: torch.Tensor,
    reg_noise_std=1.0 / 30.0,
):
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    out = per_image_dip_net(net_input)

    return out.mean(axis=0, keepdims=True)


# Per-image routine
def train_dip_on_image(
    per_image_dip_net: nn.Sequential,
    net_input,
    attacked_img,
    num_iter,
    input_depth,
    ses_parameter,
    dip_objective: EWC,
    anchoring_loss_fn,
    optimizer,
    anchoring_loss_weight=1.0,
    reg_noise_std=1.0 / 30.0,
    past_net_inputs=[],
    past_attacks=[],
    past_defenses=[],
):
    series, out_series, delta_series = [], [], []
    psnr_max_img, SES_img = None, None

    # Loss
    img_noisy_torch = np_to_torch(attacked_img).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    check_past_example_losses = len(past_net_inputs) > 0
    past_net_inputs = [
        np_to_torch(past_input).type(dtype).to(net_input.device)
        for past_input in past_net_inputs
    ]
    past_attacks = np_to_torch(np.array(past_attacks)).type(dtype).squeeze()
    past_defenses = np_to_torch(np.array(past_defenses)).type(dtype).squeeze()

    past_example_losses = []
    current_example_losses = []

    # Can probably clean the series thing up a bit more
    def closure(net_input, psnr_max_img, SES_img):
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = per_image_dip_net(net_input)
        current_attack_loss = dip_objective(
            img_noisy_torch,
            out.mean(dim=0, keepdim=True),
            list(per_image_dip_net.parameters()),
        )

        past_attacks_current_defense = None

        if check_past_example_losses:
            past_attacks_current_defense = torch.stack(
                [
                    dip_output_of_image(
                        per_image_dip_net,
                        past_input,
                        reg_noise_std,
                    )
                    for past_input in past_net_inputs
                ]
            )
            past_attacks_losses = torch.stack(
                [
                    anchoring_loss_fn(
                        past_attack.unsqueeze(0),
                        past_defense.unsqueeze(0),
                        past_attack_current_defense.unsqueeze(0),
                    )
                    for past_attack, past_defense, past_attack_current_defense in zip(
                        past_attacks,
                        past_defenses,
                        past_attacks_current_defense,
                    )
                ]
            )

            reversed_idxs = torch.flip(torch.arange(len(past_net_inputs)), dims=(0,))
            past_attack_weights = 0.99 ** (1 + reversed_idxs)
            past_attack_weights = torch.clip(past_attack_weights, min=0.9)
            past_attack_weights = past_attack_weights.type(dtype).to(net_input.device)

            past_attacks_losses = (past_attacks_losses * past_attack_weights).sum()
            past_attack_weight_sum = past_attack_weights.sum()
        else:
            past_attacks_losses = torch.tensor(0.0).type(dtype)
            past_attack_weight_sum = 1.0

        total_loss = (current_attack_loss + past_attacks_losses) / (
            1 + past_attack_weight_sum
        )
        past_example_losses.append(past_attacks_losses.item())
        current_example_losses.append(current_attack_loss.item())

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

        if psnr_max_img is None:
            psnr_max_img = out.detach().cpu().numpy().mean(axis=0)

        return (
            total_loss,
            (net_input, psnr_max_img, SES_img),
            past_attacks_current_defense,
        )

    total_losses = []
    for n_train_iter in tqdm(range(num_iter), desc="DIP Denoising iterations"):
        optimizer.zero_grad()
        (
            total_loss,
            (net_input, psnr_max_img, SES_img),
            past_attacks_current_defenses,
        ) = closure(net_input, psnr_max_img, SES_img)
        total_losses.append(total_loss)
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
            "total_losses": total_losses[::100],
            "past_example_losses": past_example_losses[::100],
            "current_example_losses": current_example_losses[::100],
            "past_defenses": past_attacks_current_defenses,
        },
    )


def main(argv):
    print("Use the PGD Attack by default for testing.")
    print("Torch", torch.__version__, "CUDA", torch.version.cuda)
    print("Device:", torch.device("cuda:0"))

    assert FLAGS.anchor_to in [
        "defense",
        "attack",
    ], f"Invalid anchor_to value: {FLAGS.anchor_to}. Must be 'defense' or 'attack'."

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
    num_iter = FLAGS.num_iter_per_image  # net_info["num_iter"]
    dip_net = dip_net.type(dtype)

    validation_set = get_imagenet_dataset("val")
    torch.manual_seed(0)
    example_dataloader = DataLoader(
        validation_set, batch_size=1, shuffle=True, drop_last=True
    )
    tk = torchattacks.PGD(resnet_classifier, eps=0.03, alpha=0.005, steps=40)

    if FLAGS.anchoring_loss_fn == "mse":
        anchoring_loss_fn = lambda attack, defense, output: mse_loss(output, defense)
    elif FLAGS.anchoring_loss_fn == "spectral":
        anchoring_loss_fn = spectral_anchoring_loss(
            FLAGS.fourier_mask_alpha, (224, 224), dev, anchor_to=FLAGS.anchor_to
        )
    elif FLAGS.anchoring_loss_fn == "fake":
        anchoring_loss_fn = fake_loss(dev)
    else:
        raise ValueError(
            f"Unsupported anchoring loss function: {FLAGS.anchoring_loss_fn}"
        )

    num_attacks_so_far = 0
    ses_parameter = FLAGS.SES_lambda
    prev_x_attacks = []
    prev_x_inputs = []
    prev_ses_imgs = []
    dip_objective = EWC(
        main_task_objective=mse_loss,
        past_task_weight=FLAGS.past_task_weight,
    ).to(dev)
    optimizer = torch.optim.Adam(get_params("net", dip_net, None), lr=0.01)

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
        # current_net_input = adv_image_input + 0.15 * np.random.randn(
        #     *adv_image_input.shape
        # )
        current_net_input = np.clip(
            0.5 + 0.5 * np.random.randn(*adv_image_input.shape), 0.0, 1.0
        )
        net_input = np_to_torch(current_net_input).type(dtype).to(dev)

        dip_defend_output = train_dip_on_image(
            per_image_dip_net=dip_net,  # This net will be updated constantly
            attacked_img=adv_image_input,
            net_input=net_input,
            num_iter=num_iter,
            input_depth=input_depth,
            ses_parameter=ses_parameter,
            dip_objective=dip_objective,
            optimizer=optimizer,
            anchoring_loss_fn=anchoring_loss_fn,
            anchoring_loss_weight=FLAGS.anchoring_loss_weight,
            past_attacks=prev_x_attacks,
            past_net_inputs=prev_x_inputs,
            past_defenses=prev_ses_imgs,
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
        past_attacks_logits = []
        with torch.no_grad():
            for idx, past_example in enumerate(prev_x_attacks):
                past_attacks_defended = dip_output_of_image(
                    dip_net,
                    np_to_torch(prev_x_inputs[idx]).type(dtype).to(dev),
                    reg_noise_std=1.0 / 30.0,
                )
                past_example_defended_logits = resnet_classifier(past_attacks_defended)
                past_attacks_logits.append(
                    past_example_defended_logits.detach().cpu().numpy()
                )

        print("Saving defense results...")
        os.makedirs(FLAGS.save_dir, exist_ok=True)
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
                    "past_attacks_logits": past_attacks_logits,
                },
                f,
            )

        num_attacks_so_far += 1
        prev_x_attacks.append(adv_image_input)
        prev_x_inputs.append(current_net_input)
        prev_ses_imgs.append(ses_img)
        if FLAGS.update_ewc:
            img_noisy_torch = np_to_torch(adv_image_input).type(dtype)
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()
            net_input = net_input_saved + (noise.normal_() * 1 / 30.0)
            out = dip_net(net_input)

            final_total_loss = (
                (out.mean(dim=0, keepdim=True) - img_noisy_torch).pow(2).mean()
            )

            dip_objective.expand_examples(list(dip_net.parameters()), final_total_loss)

        if num_attacks_so_far >= FLAGS.attack_length:
            break


if __name__ == "__main__":
    app.run(main)
