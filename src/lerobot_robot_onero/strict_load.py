"""PI05 loading that never accepts upstream's research-time random-weight fallback."""


def load_pi05_strict(policy_class, config, checkpoint):
    import torch
    from safetensors.torch import load_file

    if getattr(config, "use_proprioceptive_memory", False):
        raise ValueError("PI05 memory extensions need a separately validated checkpoint loader")
    policy = policy_class(config)
    raw = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    if not raw:
        raise ValueError("PI05 checkpoint has no tensors")
    for name, tensor in raw.items():
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"PI05 checkpoint contains non-finite tensor: {name}")
    converted = policy._convert_openpi_state_dict(raw)
    for name in raw:
        expected = name
        for old, new in (
            ("action_time_mlp_in.", "time_mlp_in."),
            ("action_time_mlp_out.", "time_mlp_out."),
        ):
            if expected.startswith(old):
                expected = new + expected[len(old) :]
        if not expected.startswith("model."):
            expected = "model." + expected
        if expected not in converted:
            raise ValueError(f"upstream PI05 conversion discarded checkpoint tensor: {name}")
    result = policy.load_state_dict(converted, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError("PI05 checkpoint did not load exactly")
    return policy
