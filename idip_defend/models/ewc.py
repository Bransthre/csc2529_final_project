from typing import Any, List

import torch
import torch.nn as nn


class EWC(nn.Module):
    def __init__(
        self,
        main_task_objective: nn.Module,
        past_task_weight: float = 1.0,
    ) -> None:
        super().__init__()
        # do something
        self.main_task_objective = main_task_objective
        self.past_tasks: List[Any, torch.Tensor] = []
        self.past_task_weight = past_task_weight

    def expand_examples(self, new_params, new_loss):
        # do something, should take the parameter and consider the fisher
        task_grads = [
            torch.autograd.grad(new_loss, new_param)[0] for new_param in new_params
        ]
        task_fisher_diags = [grad.pow(2).detach() for grad in task_grads]
        self.past_tasks.append((new_params, task_fisher_diags))

    def forward(
        self,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        input_net_params: List,
    ) -> torch.Tensor:
        subtask_losses = []
        for task_id, (task_params, task_fisher_diags) in self.past_tasks.items():
            # compute EWC loss
            subtask_loss_over_params = []
            for param, diag in zip(task_params, task_fisher_diags):
                subtask_loss_over_params.append(
                    (diag * (param - input_net_params[param]).pow(2)).sum()
                )
            subtask_losses.append(subtask_loss_over_params.sum())
        total_loss = self.main_task_objective(input_img, target_img)
        if subtask_losses:
            total_loss += self.past_task_weight * sum(subtask_losses)
        return total_loss
