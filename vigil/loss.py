import torch
import torch.nn.functional as F


def vigil_loss(
    chosen_see,
    rejected_see,
    chosen_blind,
    ref_chosen_see,
    ref_rejected_see,
    ref_chosen_blind,
    beta=0.1,
    grounding_weight=1.0,
    gate_reduction="batch",
    detach_gate=False,
):
    """Equations 1, 6–8; batch gating averages the per-example Eq. 7 gates."""
    values = (chosen_see, rejected_see, chosen_blind,
              ref_chosen_see, ref_rejected_see, ref_chosen_blind)
    if any(value.ndim != 1 or value.shape != chosen_see.shape for value in values):
        raise ValueError("All likelihoods must have the same nonempty [batch] shape")
    if not chosen_see.numel() or beta <= 0 or grounding_weight < 0:
        raise ValueError("Use a nonempty batch, beta > 0, and grounding_weight >= 0")
    if gate_reduction not in ("batch", "sample"):
        raise ValueError("gate_reduction must be 'batch' or 'sample'")
    chosen, rejected, blind = (value.float() for value in values[:3])
    ref_chosen, ref_rejected, ref_blind = (value.detach().float() for value in values[3:])
    chosen_reward = beta * (chosen - ref_chosen)
    rejected_reward = beta * (rejected - ref_rejected)
    blind_reward = beta * (blind - ref_blind)
    dpo = -F.logsigmoid(chosen_reward - rejected_reward)
    cvd = -F.logsigmoid(chosen_reward - blind_reward)
    vig = chosen - blind
    gate = 1.0 - torch.tanh(vig.abs())
    if detach_gate:
        gate = gate.detach()
    if gate_reduction == "batch":
        gate = gate.mean()
    grounding = (gate * cvd).mean()
    loss = dpo.mean() + grounding_weight * grounding
    metrics = {
        "loss/dpo": dpo.mean().detach(),
        "loss/cvd": cvd.mean().detach(),
        "loss/grounding": grounding.detach(),
        "gate": gate.mean().detach(),
        "vig": vig.mean().detach(),
        "rewards/chosen": chosen_reward.mean().detach(),
        "rewards/rejected": rejected_reward.mean().detach(),
        "rewards/accuracy": (chosen_reward > rejected_reward).float().mean().detach(),
    }
    return loss, metrics
