"""Optional source-component conditioning for fixed-anchor 2D NSOT.

IDs are sampled latent variables, not target labels. Keep each point's original
ID throughout interpolation and integration; never classify a moving point by
its nearest center. The absent/zero option retains the original model path.
"""

import torch
from torch import nn

import anchor_flow


def embedding_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("model.anchor_id_count must be an integer >= 0")
    return value


def settings(config):
    count = embedding_count(config.get("model", {}).get("anchor_id_count", 0))
    if count == 0:
        return None
    prior = anchor_flow.settings(config)
    if config.get("coupling") != "nsot" or prior is None or prior["mode"] != "anchor_prior":
        raise ValueError("anchor ID conditioning requires anchor-prior NSOT")
    if count != config.get("num_regions"):
        raise ValueError("model.anchor_id_count must equal num_regions")
    nsot = config.get("nsot", {})
    if not isinstance(nsot, dict):
        raise ValueError("nsot must be a mapping")
    if nsot.get("directional_hybrid") is not None:
        raise ValueError("anchor ID ablation requires isotropic NSOT; remove directional_hybrid")
    return {"mode": "source_component", "num_embeddings": count,
            "embedding_initialization": "zeros; unchanged backbone initialization and RNG",
            "injection": "per-point embedding added before the transformer",
            "training_ids": "original cached source-component labels at the sampled pair indices",
            "inference_ids": "original categorical labels drawn with the saved GMM source",
            "id_lifetime": "fixed for each particle throughout interpolation and every Euler step",
            "target_label_access": False, "quality_guarantee": False}


def make_embedding(count, width):
    """Add no parameters or RNG draws when disabled; no random draws when enabled."""
    count = embedding_count(count)
    if count == 0:
        return None
    return nn.Embedding.from_pretrained(torch.zeros(count, width), freeze=False)


def embedding_features(embedding, component_ids, x):
    """Validate structural invariants without a data-dependent CUDA synchronization."""
    if embedding is None:
        if component_ids is not None:
            raise ValueError("component_ids were supplied to an unconditioned model")
        return None
    if component_ids is None:
        raise ValueError("Anchor-conditioned model requires original component_ids")
    if not isinstance(component_ids, torch.Tensor) or component_ids.dtype != torch.long \
            or component_ids.shape != x.shape[:2] or component_ids.device != x.device:
        raise ValueError("component_ids must be int64 [B,N] on the same device as the points")
    if component_ids.device.type == "cpu" and component_ids.numel() \
            and ((component_ids < 0).any() or (component_ids >= embedding.num_embeddings).any()):
        raise ValueError("component_ids are outside the embedding range")
    # CUDA embedding checks indices natively; sampler IDs are range-validated at setup.
    return embedding(component_ids)


def velocity(model, x, t, component_ids=None):
    """Preserve two-argument calls for baseline and third-party diagnostic models."""
    if component_ids is None:
        return model(x, t)
    return model(x, t, component_ids=component_ids)


def sample_source_for_model(config, batch_size, *, device, dtype, generator=None):
    """Keep precisely the existing source draws, retaining IDs only when enabled."""
    if settings(config) is None:
        return anchor_flow.sample_source(config, batch_size, device=device, dtype=dtype,
                                         generator=generator), None
    return anchor_flow.sample_source_with_components(config, batch_size, device=device,
                                                     dtype=dtype, generator=generator)
