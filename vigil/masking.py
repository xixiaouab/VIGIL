import torch


def visual_token_mask(input_ids, config):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for name in ("image_token_id", "video_token_id", "image_token_index", "video_token_index"):
        token_id = getattr(config, name, None)
        if token_id is not None:
            mask |= input_ids.eq(token_id)
    return mask


def attention_masks(attention_mask, visual_mask, dtype, sliding_window=None):
    """Causal masks; the blind mask blocks every text-query / visual-key edge."""
    if attention_mask.ndim != 2 or attention_mask.shape != visual_mask.shape:
        raise ValueError("attention_mask and visual_mask must have shape [batch, sequence]")
    if not dtype.is_floating_point:
        raise ValueError("The additive attention mask requires a floating point dtype")
    length = attention_mask.shape[1]
    positions = torch.arange(length, device=attention_mask.device)
    allowed = positions[:, None] >= positions[None, :]
    if sliding_window is not None:
        allowed &= positions[:, None] - positions[None, :] < sliding_window
    allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
    see = torch.zeros(allowed.shape, device=attention_mask.device, dtype=dtype)
    see.masked_fill_(~allowed, float("-inf"))
    blocked = (~visual_mask[:, None, :, None]) & visual_mask[:, None, None, :]
    blind = see.masked_fill(blocked, float("-inf"))
    return see, blind
