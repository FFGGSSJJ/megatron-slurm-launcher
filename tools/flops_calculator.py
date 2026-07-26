#!/usr/bin/env python3
"""Parameter / FLOPs calculator for this repo's model configs.

Reads the same models/*.env files that common/train.sh sources (NUM_LAYERS,
HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE, ...) and mirrors common/model.sh for
everything the env file doesn't pin down: head dim (hidden/heads, the Megatron
kv-channels default), MLA ranks, attention type (gqa), SEQ_LEN (4096),
GBS (1024). Vocab defaults to the Apertus-8B tokenizer size; the performance
launch scripts override it to 335232, pass --vocab-size to match a run.

Sliding-window attention (ATTENTION_TYPE=swa) has exactly the same projections
as GQA, so params match; the attention score/context term is counted at full
sequence length, i.e. an upper bound that ignores WINDOW_SIZE and
WINDOW_ATTN_SKIP_FREQ.

Examples:
    tools/flops_calculator.py moe_117b_a8b_latent
    tools/flops_calculator.py chonk/chonk_swa_H3584_h2048_lt1792
    tools/flops_calculator.py moe_2b/moe_lt4_se_2b5_a440m moe_2b/moe_alt_se_2b5_a440m
    tools/flops_calculator.py moe_670b_a37b --attention-type mla --seq-len 8192 --gbs 2048
"""

import argparse
import os
import re
import sys

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "models")

# swa keeps GQA's projections, so it shares the GQA params/FLOPs path.
# kda is deliberately absent: linear-attention layers are a different shape.
ATTENTION_TYPES = ("gqa", "swa", "mla")
GQA_LIKE = ("gqa", "swa")

# MLA hyperparameters are hardcoded in common/model.sh, not in the model envs
MLA_DEFAULTS = {
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "q_nope_head_dim": 128,  # --qk-head-dim
    "q_rope_head_dim": 64,   # --qk-pos-emb-head-dim
    "v_head_dim": 128,       # --v-head-dim
}

_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)=(.*)$")


def parse_model_env(path):
    """Parse KEY=VALUE assignments from a models/*.env shell file."""
    env = {}
    with open(path) as f:
        for line in f:
            m = _ASSIGN_RE.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if val[:1] in ("'", '"'):
                end = val.find(val[0], 1)
                val = val[1:end] if end != -1 else val[1:]
            else:
                val = val.split("#", 1)[0].strip()
            # shell escapes like \( \) in MOE_LAYER_FREQ
            env[key] = val.replace("\\", "")
    return env


def parse_moe_layer_freq(value, num_layers):
    """Mirror Megatron --moe-layer-freq: an int N (MoE every N-th layer) or a
    python list expression like ([0]*3+[1]*10). Returns per-layer 0/1 flags."""
    if not value:
        return [1] * num_layers
    try:
        n = int(value)
        return [1 if i % n == 0 else 0 for i in range(num_layers)]
    except ValueError:
        pass
    pattern = eval(value, {"__builtins__": {}}, {})
    if len(pattern) != num_layers:
        raise ValueError(
            f"MOE_LAYER_FREQ pattern length {len(pattern)} != NUM_LAYERS {num_layers}: {value}"
        )
    return pattern


def parse_window_size(value):
    """WINDOW_SIZE is a Megatron --window-size tuple, e.g. (1024,0) = 1024 tokens
    of left context. Display only; returns the left window or None."""
    if not value:
        return None
    m = re.search(r"-?\d+", value)
    return int(m.group()) if m else None


def build_config(env, args):
    hidden_size = int(env["HIDDEN_SIZE"])
    num_layers = int(env["NUM_LAYERS"])
    num_heads = int(env["NUM_ATTENTION_HEADS"])
    pattern = parse_moe_layer_freq(env.get("MOE_LAYER_FREQ"), num_layers)
    shared_ffn = int(env.get("MOE_SHARED_FFN_HIDDEN_SIZE") or 0)
    attention_type = args.attention_type or env.get("ATTENTION_TYPE", "gqa")
    if attention_type not in ATTENTION_TYPES:
        sys.exit(f"Unsupported ATTENTION_TYPE {attention_type!r} "
                 f"(supported: {', '.join(ATTENTION_TYPES)}); pass --attention-type to override")

    def latent(key):
        # empty/absent means "no latent" in common/model.sh -> stays on hidden
        return int(env[key]) if env.get(key) else hidden_size

    config = {
        "model_name": env.get("MODEL_NAME", "?"),
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(env["NUM_QUERY_GROUPS"]),
        "head_dim": hidden_size // num_heads,
        "intermediate_size": int(env["FFN_HIDDEN_SIZE"]),
        "num_experts": int(env["NUM_EXPERTS"]),
        "num_experts_per_tok": int(env["TOPK"]),
        "moe_intermediate_size": int(env["MOE_FFN_HIDDEN_SIZE"]),
        # --moe-shared-expert-intermediate-size is the total shared width,
        # i.e. one fused shared expert (0 disables it)
        "shared_expert": 1 if shared_ffn > 0 else 0,
        "shared_moe_intermediate_size": shared_ffn,
        "moe_latent_size": latent("MOE_LATENT_SIZE"),
        "moe_asym_fc1_latent_size": latent("MOE_ASYMMETRIC_FC1_LATENT_SIZE"),
        "moe_asym_fc2_latent_size": latent("MOE_ASYMMETRIC_FC2_LATENT_SIZE"),
        "num_moe_layers": sum(1 for x in pattern if x),
        "num_dense_layers": sum(1 for x in pattern if not x),
        # ATTENTION_TYPE is set by the launch script, not the model env
        "attention_type": attention_type,
        "window_size": parse_window_size(env.get("WINDOW_SIZE")) if attention_type == "swa" else None,
        "vocab_size": args.vocab_size,
        "sequence_length": args.seq_len,
        "gbs": args.gbs,
    }
    config.update(MLA_DEFAULTS)
    return config


def calculate_flops(config):
    """Model FLOPs per iteration (fwd+bwd via the x3 factor)."""
    gbs = config["gbs"]
    vocab_size = config["vocab_size"]
    seq_len = config["sequence_length"]
    hidden_size = config["hidden_size"]
    head_dim = config["head_dim"]
    num_query_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    layer_num = config["num_hidden_layers"]

    num_experts = config["num_experts"]
    num_experts_per_tok = config["num_experts_per_tok"]
    moe_intermediate_size = config["moe_intermediate_size"]
    intermediate_size = config["intermediate_size"]
    shared_expert = config["shared_expert"]
    moe_latent_size = config["moe_latent_size"]
    moe_asym_fc1_latent_size = config["moe_asym_fc1_latent_size"]
    moe_asym_fc2_latent_size = config["moe_asym_fc2_latent_size"]
    shared_moe_intermediate_size = config["shared_moe_intermediate_size"]

    model_flops = 0
    token_num = gbs * seq_len

    # embedding flops
    embd_proj = 2 * token_num * hidden_size * vocab_size
    model_flops += embd_proj

    # attention flops
    if config["attention_type"] == "mla":
        # MLA (Multi-Latent Attention) - DeepSeek-V2/V3 style
        kv_lora_rank = config["kv_lora_rank"]
        q_lora_rank = config["q_lora_rank"]
        q_nope_head_dim = config["q_nope_head_dim"]
        q_rope_head_dim = config["q_rope_head_dim"]
        v_head_dim = config["v_head_dim"]

        # Query projection (optionally compressed)
        if q_lora_rank > 0:
            # Compressed: hidden -> q_lora_rank -> num_heads * (q_nope + q_rope)
            q_proj = 2 * token_num * hidden_size * q_lora_rank
            q_proj += 2 * token_num * q_lora_rank * num_query_heads * (q_nope_head_dim + q_rope_head_dim)
        else:
            # Direct: hidden -> num_heads * (q_nope + q_rope)
            q_proj = 2 * token_num * hidden_size * num_query_heads * (q_nope_head_dim + q_rope_head_dim)

        # KV compression: hidden -> kv_lora_rank (stored in cache)
        kv_down_proj = 2 * token_num * hidden_size * kv_lora_rank

        # KV up-projection during attention: kv_lora_rank -> num_heads * (q_nope + v_head_dim)
        # Note: RoPE part uses q_rope_head_dim directly from cached k_rope
        kv_up_proj = 2 * token_num * kv_lora_rank * num_query_heads * (q_nope_head_dim + v_head_dim)

        # Attention computation
        head_dim_total = q_nope_head_dim + q_rope_head_dim
        attn = 2 * gbs * seq_len * num_query_heads * head_dim_total * seq_len

        # Output projection
        output_proj = 2 * token_num * hidden_size * num_query_heads * v_head_dim

        model_flops += layer_num * (q_proj + kv_down_proj + kv_up_proj + attn + output_proj)
    else:
        # GQA (Grouped Query Attention) - standard. swa shares this path: same
        # projections, and the attn term stays at full seq_len (upper bound).
        q_proj = 2 * token_num * hidden_size * (num_query_heads * head_dim)
        kv_proj = 2 * (2 * token_num * hidden_size * (num_key_value_heads * head_dim))
        attn = 2 * gbs * seq_len * (num_query_heads * head_dim) * seq_len
        output_proj = 2 * token_num * hidden_size * (num_query_heads * head_dim)
        model_flops += layer_num * (q_proj + kv_proj + attn + output_proj)

    # MLP flops: dense/MoE split follows the MOE_LAYER_FREQ pattern
    dense_layer_num = config["num_dense_layers"]
    moe_layer_num = config["num_moe_layers"]

    # Dense MLP (SwiGLU: gate_proj + up_proj + down_proj)
    if dense_layer_num > 0:
        dense_mlp_1 = 2 * token_num * hidden_size * (intermediate_size * 2)  # gate + up
        dense_mlp_2 = 2 * token_num * intermediate_size * hidden_size  # down
        model_flops += dense_layer_num * (dense_mlp_1 + dense_mlp_2)

    # MoE MLP with optional latent projection
    if moe_layer_num > 0:
        use_moe_latent = moe_latent_size != hidden_size
        use_asym_moe_latent = (moe_asym_fc1_latent_size != hidden_size) or (moe_asym_fc2_latent_size != hidden_size)

        # Gate/router FLOPs (always present for MoE)
        gate_flops = 2 * token_num * hidden_size * num_experts

        if use_asym_moe_latent:
            # Asymmetric latent MoE: routed experts read from an fc1 latent and write to an fc2 latent.
            # A shared compress (hidden -> fc1_latent) and decompress (fc2_latent -> hidden) run once
            # per token (shared across experts) and only exist when that side is actually compressed.
            # The shared expert stays on hidden. Setting fc1_latent == hidden reduces this to an
            # output-only latent; both == hidden reduces to a plain MoE.
            moe_compress = 2 * token_num * hidden_size * moe_asym_fc1_latent_size if moe_asym_fc1_latent_size != hidden_size else 0
            moe_decompress = 2 * token_num * moe_asym_fc2_latent_size * hidden_size if moe_asym_fc2_latent_size != hidden_size else 0
            routed_mlp_1 = num_experts_per_tok * 2 * token_num * moe_asym_fc1_latent_size * (moe_intermediate_size * 2)
            routed_mlp_2 = num_experts_per_tok * 2 * token_num * moe_intermediate_size * moe_asym_fc2_latent_size
            shared_mlp_1 = shared_expert * 2 * token_num * hidden_size * (shared_moe_intermediate_size * 2)
            shared_mlp_2 = shared_expert * 2 * token_num * shared_moe_intermediate_size * hidden_size
            model_flops += moe_layer_num * (gate_flops + moe_compress + routed_mlp_1 + routed_mlp_2 + shared_mlp_1 + shared_mlp_2 + moe_decompress)
        elif use_moe_latent:
            moe_compress = 2 * token_num * hidden_size * moe_latent_size
            routed_mlp_1 = num_experts_per_tok * 2 * token_num * moe_latent_size * (moe_intermediate_size * 2)
            routed_mlp_2 = num_experts_per_tok * 2 * token_num * moe_intermediate_size * moe_latent_size
            shared_mlp_1 = shared_expert * 2 * token_num * moe_latent_size * (shared_moe_intermediate_size * 2)
            shared_mlp_2 = shared_expert * 2 * token_num * shared_moe_intermediate_size * moe_latent_size
            moe_decompress = 2 * token_num * moe_latent_size * hidden_size
            model_flops += moe_layer_num * (gate_flops + moe_compress + routed_mlp_1 + routed_mlp_2 + shared_mlp_1 + shared_mlp_2 + moe_decompress)
        else:
            routed_mlp_1 = num_experts_per_tok * 2 * token_num * hidden_size * (moe_intermediate_size * 2)
            routed_mlp_2 = num_experts_per_tok * 2 * token_num * moe_intermediate_size * hidden_size
            shared_mlp_1 = shared_expert * 2 * token_num * hidden_size * (shared_moe_intermediate_size * 2)
            shared_mlp_2 = shared_expert * 2 * token_num * shared_moe_intermediate_size * hidden_size
            model_flops += moe_layer_num * (gate_flops + routed_mlp_1 + routed_mlp_2 + shared_mlp_1 + shared_mlp_2)

    model_flops_t = model_flops * 3 / 1e12
    non_embd_model_flops_t = (model_flops - embd_proj) * 3 / 1e12
    non_embd_model_flops_per_token = non_embd_model_flops_t / token_num

    print(
        f"Model FLOPs: {model_flops_t:.2f}T | Non-embd FLOPs: {non_embd_model_flops_t:.2f}T"
        f" | Non-embd FLOPs per token: {non_embd_model_flops_per_token*1e3:.2f}B"
    )
    return model_flops


def calculate_model_size(config):
    """
    Calculate total model parameters and activated parameters per token.
    Returns (total_params, activated_params) in billions.
    """
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    head_dim = config["head_dim"]
    num_query_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    layer_num = config["num_hidden_layers"]

    num_experts = config["num_experts"]
    num_experts_per_tok = config["num_experts_per_tok"]
    moe_intermediate_size = config["moe_intermediate_size"]
    intermediate_size = config["intermediate_size"]
    shared_expert = config["shared_expert"]
    moe_latent_size = config["moe_latent_size"]
    moe_asym_fc1_latent_size = config["moe_asym_fc1_latent_size"]
    moe_asym_fc2_latent_size = config["moe_asym_fc2_latent_size"]
    shared_moe_intermediate_size = config["shared_moe_intermediate_size"]

    total_params = 0
    activated_params = 0

    # Embedding layers: model.sh passes --untie-embeddings-and-output-weights
    embd_params = 2 * vocab_size * hidden_size
    total_params += embd_params
    activated_params += embd_params

    # Attention parameters per layer
    if config["attention_type"] == "mla":
        # MLA (Multi-Latent Attention)
        kv_lora_rank = config["kv_lora_rank"]
        q_lora_rank = config["q_lora_rank"]
        q_nope_head_dim = config["q_nope_head_dim"]
        q_rope_head_dim = config["q_rope_head_dim"]
        v_head_dim = config["v_head_dim"]

        if q_lora_rank > 0:
            # Compressed query: hidden -> q_lora_rank -> num_heads * (q_nope + q_rope)
            q_params = hidden_size * q_lora_rank + q_lora_rank * num_query_heads * (q_nope_head_dim + q_rope_head_dim)
        else:
            # Direct query
            q_params = hidden_size * num_query_heads * (q_nope_head_dim + q_rope_head_dim)

        # KV compression: hidden -> kv_lora_rank
        kv_down_params = hidden_size * kv_lora_rank
        # KV up: kv_lora_rank -> num_heads * (q_nope + v_head_dim)
        kv_up_params = kv_lora_rank * num_query_heads * (q_nope_head_dim + v_head_dim)
        # Output projection
        o_params = num_query_heads * v_head_dim * hidden_size

        attn_params_per_layer = q_params + kv_down_params + kv_up_params + o_params
    else:
        # GQA (Grouped Query Attention); swa is identical parameter-wise
        q_params = hidden_size * (num_query_heads * head_dim)
        k_params = hidden_size * (num_key_value_heads * head_dim)
        v_params = hidden_size * (num_key_value_heads * head_dim)
        o_params = (num_query_heads * head_dim) * hidden_size
        attn_params_per_layer = q_params + k_params + v_params + o_params

    total_params += layer_num * attn_params_per_layer
    activated_params += layer_num * attn_params_per_layer

    # MLP parameters: dense/MoE split follows the MOE_LAYER_FREQ pattern
    dense_layer_num = config["num_dense_layers"]
    moe_layer_num = config["num_moe_layers"]

    # Dense MLP (SwiGLU: gate_proj, up_proj, down_proj)
    if dense_layer_num > 0:
        dense_mlp_params = 3 * hidden_size * intermediate_size
        total_params += dense_layer_num * dense_mlp_params
        activated_params += dense_layer_num * dense_mlp_params

    # MoE MLP
    if moe_layer_num > 0 and num_experts > 1:
        # MoE architecture
        use_moe_latent = moe_latent_size != hidden_size
        use_asym_moe_latent = (moe_asym_fc1_latent_size != hidden_size) or (moe_asym_fc2_latent_size != hidden_size)

        # Gate/router (operates on hidden_size)
        gate_params = hidden_size * num_experts
        total_params += moe_layer_num * gate_params
        activated_params += moe_layer_num * gate_params

        if use_asym_moe_latent:
            # Shared compress (hidden -> fc1_latent), only when the fc1 side is compressed
            if moe_asym_fc1_latent_size != hidden_size:
                compress_params = hidden_size * moe_asym_fc1_latent_size
                total_params += moe_layer_num * compress_params
                activated_params += moe_layer_num * compress_params

            # Routed expert: gate + up read from fc1_latent, down writes to fc2_latent
            expert_params = 2 * moe_asym_fc1_latent_size * moe_intermediate_size + moe_intermediate_size * moe_asym_fc2_latent_size
            total_routed_experts_params = num_experts * expert_params
            total_params += moe_layer_num * total_routed_experts_params

            activated_routed_params = num_experts_per_tok * expert_params
            activated_params += moe_layer_num * activated_routed_params

            if shared_expert > 0:
                shared_expert_params = 3 * hidden_size * shared_moe_intermediate_size
                total_params += moe_layer_num * shared_expert_params * shared_expert
                activated_params += moe_layer_num * shared_expert_params * shared_expert

            # Shared decompress (fc2_latent -> hidden), only when the fc2 side is compressed
            if moe_asym_fc2_latent_size != hidden_size:
                decompress_params = moe_asym_fc2_latent_size * hidden_size
                total_params += moe_layer_num * decompress_params
                activated_params += moe_layer_num * decompress_params
        elif use_moe_latent:
            compress_params = hidden_size * moe_latent_size
            total_params += moe_layer_num * compress_params
            activated_params += moe_layer_num * compress_params

            expert_params = 3 * moe_latent_size * moe_intermediate_size
            total_routed_experts_params = num_experts * expert_params
            total_params += moe_layer_num * total_routed_experts_params

            activated_routed_params = num_experts_per_tok * expert_params
            activated_params += moe_layer_num * activated_routed_params

            if shared_expert > 0:
                shared_expert_params = 3 * hidden_size * shared_moe_intermediate_size
                total_params += moe_layer_num * shared_expert_params * shared_expert
                activated_params += moe_layer_num * shared_expert_params * shared_expert

            decompress_params = moe_latent_size * hidden_size
            total_params += moe_layer_num * decompress_params
            activated_params += moe_layer_num * decompress_params
        else:
            expert_params = 3 * hidden_size * moe_intermediate_size
            total_routed_experts_params = num_experts * expert_params
            total_params += moe_layer_num * total_routed_experts_params

            activated_routed_params = num_experts_per_tok * expert_params
            activated_params += moe_layer_num * activated_routed_params

            if shared_expert > 0:
                shared_expert_params = 3 * hidden_size * shared_moe_intermediate_size
                total_params += moe_layer_num * shared_expert_params * shared_expert
                activated_params += moe_layer_num * shared_expert_params * shared_expert
    elif moe_layer_num > 0:
        # All-MoE model with num_experts <= 1 treated as dense
        dense_mlp_params = 3 * hidden_size * intermediate_size
        total_params += moe_layer_num * dense_mlp_params
        activated_params += moe_layer_num * dense_mlp_params

    # Convert to billions
    total_params_b = total_params / 1e9
    activated_params_b = activated_params / 1e9
    non_embd_activated_params_b = activated_params_b - embd_params / 1e9
    embd_params_b = embd_params / 1e9
    sparsity = non_embd_activated_params_b / total_params_b

    print(f"Total Parameters: {total_params_b:.2f}B | Activated Parameters: {activated_params_b:.2f}B | Non-Embd Activated Parameters: {non_embd_activated_params_b:.2f}B | Sparsity (non-embd): {sparsity*100:.2f}% | Activated Ratio: {num_experts_per_tok/num_experts*100:.2f}%")

    return (total_params_b, activated_params_b, non_embd_activated_params_b, embd_params_b)


def resolve_model_env(name):
    """Accept a path to a .env file or a MODEL_ENV name (resolved against
    models/, same as common/train.sh)."""
    if os.path.isfile(name):
        return name
    candidate = os.path.join(MODELS_DIR, name + ".env")
    if os.path.isfile(candidate):
        return candidate
    sys.exit(f"Model env not found: {name} (looked for {candidate})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="+",
                        help="MODEL_ENV name (e.g. moe_117b_a8b_latent, moe_2b/moe_se_2b5_a440m) or path to a models/*.env file")
    parser.add_argument("--seq-len", type=int, default=4096, help="sequence length (default: %(default)s, per common/model.sh)")
    parser.add_argument("--gbs", type=int, default=1024, help="global batch size (default: %(default)s, per common/model.sh)")
    parser.add_argument("--vocab-size", type=int, default=131072,
                        help="vocab size (default: %(default)s ~ Apertus-8B tokenizer; performance launches use 335232)")
    parser.add_argument("--attention-type", choices=list(ATTENTION_TYPES), default=None,
                        help="override attention type (default: env file's ATTENTION_TYPE or gqa); "
                             "swa is counted as gqa with full-length attention")
    args = parser.parse_args()

    for name in args.models:
        env_file = resolve_model_env(name)
        config = build_config(parse_model_env(env_file), args)
        env_label = os.path.relpath(env_file, os.path.dirname(MODELS_DIR))
        attn_label = config["attention_type"]
        if config["window_size"]:
            attn_label += f" (window={config['window_size']}, counted as full attention)"
        print(f"== {config['model_name']} ({env_label}) | "
              f"{attn_label} | seq_len={config['sequence_length']} "
              f"gbs={config['gbs']} vocab={config['vocab_size']} | "
              f"{config['num_dense_layers']} dense + {config['num_moe_layers']} moe layers ==")
        calculate_model_size(config)
        calculate_flops(config)
        print()


if __name__ == "__main__":
    main()
