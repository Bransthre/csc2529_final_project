from typing import Any, List

import torch
import torch.nn as nn


class EWC(nn.Module):
    "Elastic Weight Consolidation (EWC) regularization module."

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

    def _detach_and_clone(self, lst):
        return [item.detach().clone() for item in lst]

    def expand_examples(self, prev_task_params, prev_task_loss):
        """
        Store the parameters and fisher diagonals for a previous task

        Args:
            prev_task_params: List of torch.Tensor
                The parameters of the network after training on the previous task
            prev_task_loss: torch.Tensor
                The loss on the previous task, used to compute gradients for fisher
        """
        task_grads = [
            torch.autograd.grad(prev_task_loss, new_param, retain_graph=True)[0]
            for new_param in prev_task_params
        ]

        task_fisher_diags = [
            torch.tensor(1.0).to(prev_task_params[0].device) for _ in task_grads
        ]

        self.past_tasks.append(
            (
                self._detach_and_clone(prev_task_params),
                self._detach_and_clone(task_fisher_diags),
            )
        )

    def forward(
        self,
        input_img: torch.Tensor,
        target_img: torch.Tensor,
        input_net_params: List,
    ) -> torch.Tensor:
        task_objective = self.main_task_objective(input_img, target_img)
        if not self.past_tasks:
            return task_objective

        subtask_loss = torch.tensor(0.0, device=task_objective.device)
        num_attacks_so_far = len(self.past_tasks)
        for task_params, task_fisher_diags in self.past_tasks:
            for input_param, task_param, diag in zip(
                input_net_params, task_params, task_fisher_diags
            ):
                subtask_loss += (diag * (input_param - task_param).pow(2)).sum()

        return task_objective + self.past_task_weight * subtask_loss
