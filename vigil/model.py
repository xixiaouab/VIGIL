from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          LlavaOnevisionForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)

from .masking import attention_masks, visual_token_mask


def sequence_log_probs(logits, labels):
    labels = labels[:, 1:].to(logits.device)
    valid = labels.ne(-100)
    targets = labels.masked_fill(~valid, 0)
    token_logps = logits[:, :-1].float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logps.masked_fill(~valid, 0).sum(-1)


def response_log_probs(hidden_states, labels, lm_head, chunk_size=128):
    labels = labels[:, 1:].to(hidden_states.device)
    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError("Every sample must contain at least one response target after the first token")
    batch_indices = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)[valid]
    states = hidden_states[:, :-1][valid]
    targets = labels[valid]

    def score(state, target):
        logits = lm_head(state).float()
        return -F.cross_entropy(logits, target.to(logits.device), reduction="none").to(state.device)

    pieces = []
    for start in range(0, states.shape[0], chunk_size):
        state, target = states[start:start + chunk_size], targets[start:start + chunk_size]
        if torch.is_grad_enabled() and state.requires_grad:
            pieces.append(checkpoint(score, state, target, use_reentrant=False))
        else:
            pieces.append(score(state, target))
    return torch.zeros(labels.shape[0], device=states.device, dtype=torch.float32).scatter_add(
        0, batch_indices, torch.cat(pieces))


@dataclass
class PreparedInputs:
    embeddings: torch.Tensor
    positions: torch.Tensor
    attention_mask: torch.Tensor
    visual_mask: torch.Tensor
    labels: torch.Tensor


class VIGILModel(nn.Module):
    def __init__(self, model, logprob_chunk_size=128):
        super().__init__()
        if type(model.config).model_type not in ("qwen2_5_vl", "llava_onevision", "internvl_chat"):
            raise ValueError("Supported model types: qwen2_5_vl, llava_onevision, internvl_chat")
        if logprob_chunk_size < 1:
            raise ValueError("logprob_chunk_size must be positive")
        self.model = model
        self.logprob_chunk_size = logprob_chunk_size
        self.model.config.use_cache = False
        if self.model_type == "internvl_chat":
            self.model.config.hidden_size = self.model.config.llm_config.hidden_size
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
        decoder_config = self.text_config
        implementation = getattr(decoder_config, "attn_implementation", decoder_config._attn_implementation)
        if implementation not in ("sdpa", "eager"):
            raise ValueError("VIGIL requires sdpa or eager attention for the 4-D counterfactual mask")
        if self.model_type == "internvl_chat":
            if self.text_config.model_type != "internlm2":
                raise ValueError("The InternVL adapter targets the InternLM2-based InternVL2.5-26B checkpoint")
            decoder = self.decoder
            original = decoder._prepare_decoder_attention_mask

            def prepare_mask(instance, attention_mask, input_shape, inputs_embeds, past_key_values_length):
                if attention_mask is not None and attention_mask.ndim == 4:
                    return attention_mask
                return original(attention_mask, input_shape, inputs_embeds, past_key_values_length)

            decoder._prepare_decoder_attention_mask = MethodType(prepare_mask, decoder)

    @classmethod
    def from_pretrained(cls, name, dtype="bfloat16", attn_implementation="sdpa", **kwargs):
        chunk_size = kwargs.pop("logprob_chunk_size", 128)
        config_keys = ("revision", "cache_dir", "local_files_only", "token", "trust_remote_code")
        config = AutoConfig.from_pretrained(name, **{key: kwargs[key] for key in config_keys if key in kwargs})
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        if type(config).model_type == "internvl_chat":
            if not kwargs.get("trust_remote_code", False):
                raise ValueError("InternVL2.5 requires trust_remote_code=True")
            tokenizer = AutoTokenizer.from_pretrained(
                name, **{key: kwargs[key] for key in config_keys if key in kwargs})
            config.image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            model = AutoModel.from_pretrained(name, config=config, dtype=dtype,
                                               attn_implementation="eager", use_flash_attn=False, **kwargs)
            model.img_context_token_id = config.image_token_id
            return cls(model, logprob_chunk_size=chunk_size)
        model_class = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
                       "llava_onevision": LlavaOnevisionForConditionalGeneration}.get(type(config).model_type)
        if model_class is None:
            raise ValueError(f"Unsupported model type: {type(config).model_type}")
        model = model_class.from_pretrained(name, config=config, dtype=dtype,
                                           attn_implementation=attn_implementation, **kwargs)
        return cls(model, logprob_chunk_size=chunk_size)

    @property
    def config(self):
        return self.model.config

    @property
    def is_gradient_checkpointing(self):
        return self.model.is_gradient_checkpointing

    @property
    def _keys_to_ignore_on_save(self):
        return self.model._keys_to_ignore_on_save

    def tie_weights(self):
        return self.model.tie_weights()

    @property
    def model_type(self):
        return type(self.config).model_type

    @property
    def text_config(self):
        return self.config.llm_config if self.model_type == "internvl_chat" else self.config.text_config

    @property
    def decoder(self):
        if self.model_type == "internvl_chat":
            return self.model.language_model.model
        return self.model.model.language_model

    @property
    def lm_head(self):
        if self.model_type == "internvl_chat":
            return self.model.language_model.get_output_embeddings()
        return self.model.lm_head

    def get_input_embeddings(self):
        if self.model_type == "internvl_chat":
            return self.model.language_model.get_input_embeddings()
        return self.model.get_input_embeddings()

    def save_pretrained(self, path, **kwargs):
        return self.model.save_pretrained(path, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        options = {"use_reentrant": False}
        if gradient_checkpointing_kwargs:
            options.update(gradient_checkpointing_kwargs)
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=options)

    def _features(self, inputs):
        if inputs.get("pixel_values_videos") is not None:
            raise ValueError("The preference pipeline expects images; video batches are not supported")
        if inputs.get("pixel_values") is None:
            return None
        if self.model_type == "internvl_chat":
            pixels = inputs["pixel_values"]
            dtype = next(self.model.vision_model.parameters()).dtype
            features = self.model.extract_feature(pixels.to(dtype=dtype))
            if "image_flags" in inputs:
                features = features[inputs["image_flags"].reshape(-1).eq(1)]
            return features.reshape(-1, features.shape[-1])
        base = self.model.model
        if self.model_type == "qwen2_5_vl":
            features = base.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        else:
            features = base.get_image_features(inputs["pixel_values"], inputs["image_sizes"],
                                               batch_num_images=inputs.get("batch_num_images"))
        return torch.cat(features, dim=0)

    def prepare(self, inputs, features=None):
        ids = inputs["input_ids"]
        embeddings = self.get_input_embeddings()(ids)
        padding = inputs.get("attention_mask", torch.ones_like(ids)).to(embeddings.device)
        visual = visual_token_mask(ids, self.config).to(embeddings.device)
        if features is not None:
            features = features.to(embeddings.device, embeddings.dtype)
            if self.model_type == "internvl_chat":
                if visual.sum().item() != features.shape[0]:
                    raise ValueError("Visual-token count does not match the projected image features")
                mask = visual.unsqueeze(-1).expand_as(embeddings)
            else:
                mask, _ = self.model.model.get_placeholder_mask(ids, embeddings, image_features=features)
            embeddings = embeddings.masked_scatter(mask, features)
        elif visual.any():
            raise ValueError("Image tokens require pixel_values")
        if "position_ids" in inputs:
            positions = inputs["position_ids"]
        elif self.model_type == "qwen2_5_vl":
            positions, _ = self.model.model.get_rope_index(ids, image_grid_thw=inputs.get("image_grid_thw"),
                                                          attention_mask=padding)
        else:
            positions = padding.long().cumsum(-1) - 1
            positions.masked_fill_(padding.eq(0), 0)
        labels = inputs["labels"].to(embeddings.device)
        if labels.shape != ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        labels = labels.masked_fill(~padding.bool() | visual, -100)
        return PreparedInputs(embeddings, positions.to(embeddings.device), padding, visual, labels)

    def _score(self, prepared, blind=False):
        see_mask, blind_mask = attention_masks(prepared.attention_mask, prepared.visual_mask,
                                               prepared.embeddings.dtype)
        masks = {"full_attention": blind_mask if blind else see_mask}
        text_config = self.text_config
        if "sliding_attention" in getattr(text_config, "layer_types", []):
            see_sliding, blind_sliding = attention_masks(
                prepared.attention_mask, prepared.visual_mask, prepared.embeddings.dtype,
                sliding_window=text_config.sliding_window,
            )
            masks["sliding_attention"] = blind_sliding if blind else see_sliding
        selected_mask = masks["full_attention"] if self.model_type == "internvl_chat" else masks
        if self.model_type == "internvl_chat":
            empty_rows = torch.isneginf(selected_mask).all(-1, keepdim=True)
            selected_mask = selected_mask.masked_fill(empty_rows, 0)
        output = self.decoder(inputs_embeds=prepared.embeddings, position_ids=prepared.positions,
                              attention_mask=selected_mask, use_cache=False, return_dict=True)
        return response_log_probs(output.last_hidden_state, prepared.labels, self.lm_head,
                                  self.logprob_chunk_size)

    def forward(self, batch=None, chosen=None, rejected=None, **kwargs):
        if batch is not None:
            chosen, rejected = batch["chosen"], batch["rejected"]
        if chosen is None or rejected is None:
            raise ValueError("Provide chosen and rejected processor batches with response-only labels")
        features = self._features(chosen)
        chosen_inputs = self.prepare(chosen, features)
        for key in ("pixel_values", "image_grid_thw", "image_sizes", "batch_num_images", "image_flags"):
            if key in chosen or key in rejected:
                if key not in chosen or key not in rejected or not torch.equal(chosen[key], rejected[key]):
                    raise ValueError(f"Chosen and rejected responses must use identical {key}")
        rejected_inputs = self.prepare(rejected, features)
        return {"chosen_see": self._score(chosen_inputs),
                "rejected_see": self._score(rejected_inputs),
                "chosen_blind": self._score(chosen_inputs, blind=True)}

    @torch.no_grad()
    def generate(self, **inputs):
        if self.model_type != "internvl_chat":
            inputs.setdefault("use_cache", True)
            return self.model.generate(**inputs)
        maximum = inputs.pop("max_new_tokens", 128)
        do_sample = inputs.pop("do_sample", False)
        temperature = inputs.pop("temperature", 1.0)
        top_p, top_k = inputs.pop("top_p", 1.0), inputs.pop("top_k", 0)
        eos = inputs.pop("eos_token_id", self.text_config.eos_token_id)
        pad = inputs.pop("pad_token_id", self.text_config.pad_token_id)
        inputs.pop("use_cache", None)
        if maximum < 1 or temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Invalid generation length or sampling parameters")
        allowed = {"input_ids", "attention_mask", "pixel_values", "image_flags", "position_ids"}
        if set(inputs) - allowed:
            raise ValueError(f"Unsupported InternVL generation arguments: {sorted(set(inputs) - allowed)}")
        ids = inputs["input_ids"]
        prepared = self.prepare({**inputs, "labels": torch.full_like(ids, -100)}, self._features(inputs))
        embeddings, mask, visual = prepared.embeddings, prepared.attention_mask, prepared.visual_mask
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        eos_ids = torch.as_tensor(eos if isinstance(eos, (list, tuple)) else [eos], device=ids.device)
        pad = pad if pad is not None else int(eos_ids[0])
        for step in range(maximum):
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask.eq(0), 0)
            causal, _ = attention_masks(mask, visual, embeddings.dtype)
            causal = causal.masked_fill(torch.isneginf(causal).all(-1, keepdim=True), 0)
            output = self.decoder(inputs_embeds=embeddings, attention_mask=causal,
                                  position_ids=positions, use_cache=False, return_dict=True)
            last = (torch.arange(mask.shape[1], device=mask.device)[None] * mask).max(-1).values.long()
            states = output.last_hidden_state[torch.arange(ids.shape[0], device=last.device), last]
            logits = self.lm_head(states).float().to(ids.device)
            if do_sample:
                logits = logits / temperature
                if top_k:
                    threshold = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                if top_p < 1:
                    sorted_logits, order = logits.sort(descending=True, dim=-1)
                    remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits = logits.masked_fill(torch.zeros_like(remove).scatter(1, order, remove), float("-inf"))
                token = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            else:
                token = logits.argmax(-1)
            token = token.masked_fill(finished, pad)
            ids = torch.cat((ids, token[:, None]), dim=1)
            was_finished = finished
            finished = finished | (token[:, None] == eos_ids[None]).any(-1)
            if finished.all():
                break
            embeddings = torch.cat((embeddings, self.get_input_embeddings()(token[:, None])), dim=1)
            mask = torch.cat((mask, (~was_finished).to(mask.dtype)[:, None]), dim=1)
            visual = torch.cat((visual, torch.zeros_like(finished[:, None])), dim=1)
        return ids
