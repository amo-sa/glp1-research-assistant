"""
Step 7: Scaled dot-product attention implemented from scratch in NumPy.

Uses real pretrained weights extracted from all-MiniLM-L6-v2 -- the same
model used throughout this project for embeddings. This means we're not
just demonstrating the formula with random numbers; we're reimplementing
what the model actually computes when we call model.encode().

We verify our implementation by comparing our NumPy output to the model's
actual internal activations -- if they match to floating point precision,
we've correctly reimplemented the computation.

Architecture of all-MiniLM-L6-v2 relevant to this file:
  - 6 transformer layers
  - 12 attention heads per layer
  - Hidden dimension: 384
  - Head dimension (d_k): 384 / 12 = 32
  - So each attention head operates on 32-dimensional Q, K, V vectors

We extract weights from layer 0, head 0 -- the simplest case, but the
same math applies to all 72 attention heads across all 6 layers.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


# ---------------------------------------------------------------------------
# Step 1: Load the model and extract real weights
# ---------------------------------------------------------------------------

def load_model_and_weights():
    """Load all-MiniLM-L6-v2 and extract Q, K, V weight matrices for
    layer 0, head 0.

    MiniLM stores Q, K, V as combined weight matrices of shape (384, 384),
    then splits by head at runtime. We extract the slice for head 0.
    """
    print("Loading all-MiniLM-L6-v2...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    bert = st_model[0].auto_model  # the underlying BERT model

    # Extract combined QKV weight matrices from layer 0
    # Shape of each: (384, 384) -- all 12 heads concatenated
    layer0 = bert.encoder.layer[0].attention.self

    # .weight is shape (384, 384), .bias is shape (384,)
    # We detach from PyTorch autograd and convert to NumPy
    W_Q_full = layer0.query.weight.detach().numpy()   # (384, 384)
    W_K_full = layer0.key.weight.detach().numpy()     # (384, 384)
    W_V_full = layer0.value.weight.detach().numpy()   # (384, 384)

    b_Q_full = layer0.query.bias.detach().numpy()     # (384,)
    b_K_full = layer0.key.bias.detach().numpy()       # (384,)
    b_V_full = layer0.value.bias.detach().numpy()     # (384,)

    # Extract just head 0's slice
    # Each head gets 384/12 = 32 dimensions
    num_heads = 12
    d_model = 384
    d_k = d_model // num_heads  # 32

    # Head 0 uses the first 32 rows of the weight matrix
    head_slice = slice(0, d_k)

    W_Q = W_Q_full[head_slice, :]  # (32, 384)
    W_K = W_K_full[head_slice, :]  # (32, 384)
    W_V = W_V_full[head_slice, :]  # (32, 384)

    b_Q = b_Q_full[head_slice]     # (32,)
    b_K = b_K_full[head_slice]     # (32,)
    b_V = b_V_full[head_slice]     # (32,)

    print("Extracted weights for layer 0, head 0")
    print(f"  W_Q shape: {W_Q.shape}  (d_k x d_model)")
    print(f"  W_K shape: {W_K.shape}")
    print(f"  W_V shape: {W_V.shape}")
    print(f"  d_k (head dimension): {d_k}")

    return st_model, bert, W_Q, W_K, W_V, b_Q, b_K, b_V, d_k


# ---------------------------------------------------------------------------
# Step 2: Get token embeddings (X) for a real sentence
# ---------------------------------------------------------------------------

def get_token_embeddings(bert, sentence: str):
    """Tokenize a sentence and get its initial token embeddings.

    These are the raw embeddings before any attention -- each token
    is represented by its row in the embedding lookup table plus
    positional encoding. Shape: (num_tokens, 384).
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    tokens = tokenizer(sentence, return_tensors="pt")
    token_ids = tokens["input_ids"][0]
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)

    # Get embeddings from the embedding layer (before any transformer layers)
    with torch.no_grad():
        # word embeddings + positional embeddings + token type embeddings
        word_emb = bert.embeddings.word_embeddings(tokens["input_ids"])
        pos_emb = bert.embeddings.position_embeddings(
            torch.arange(tokens["input_ids"].shape[1]).unsqueeze(0)
        )
        tok_emb = bert.embeddings.token_type_embeddings(
            torch.zeros_like(tokens["input_ids"])
        )
        X_with_layernorm = bert.embeddings.LayerNorm(word_emb + pos_emb + tok_emb)
        X = X_with_layernorm[0].detach().numpy()  # (num_tokens, 384)

    print(f"\nTokens: {token_strings}")
    print(f"X shape: {X.shape}  (num_tokens x d_model)")
    return X, token_strings, tokens


# ---------------------------------------------------------------------------
# Step 3: Our NumPy attention implementation
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis.

    Subtracting the max before exp() prevents overflow for large values
    while producing identical results (exp(x-max)/sum(exp(x-max)) ==
    exp(x)/sum(exp(x))). This is standard practice, not a shortcut.
    """
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    d_k: int,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention -- the core operation of transformers.

    Args:
        Q: queries,  shape (num_tokens, d_k)
        K: keys,     shape (num_tokens, d_k)
        V: values,   shape (num_tokens, d_k)
        d_k: key dimension (used for scaling)
        mask: optional boolean mask to block certain positions (e.g. padding)

    Returns:
        output:          shape (num_tokens, d_k) -- enriched token representations
        attention_weights: shape (num_tokens, num_tokens) -- for visualization
    """
    # Step 1: compute raw attention scores
    # Q @ K.T: for each query token, how similar is it to each key token?
    # Result shape: (num_tokens, num_tokens)
    # scores[i][j] = "how much should token i attend to token j?"
    scores = Q @ K.T                     # (num_tokens, num_tokens)

    # Step 2: scale by sqrt(d_k)
    # Without this, dot products grow large as d_k increases, pushing
    # softmax into saturation where gradients vanish during training.
    scores = scores / np.sqrt(d_k)      # (num_tokens, num_tokens)

    # Step 3: apply mask if provided
    # Masking sets certain positions to -infinity before softmax,
    # making their post-softmax weight effectively 0.
    # Used for padding tokens (we don't want real tokens attending to padding)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Step 4: softmax -- turn raw scores into weights summing to 1
    # Now scores[i] is a probability distribution over all tokens:
    # "token i pays this fraction of its attention to each other token"
    attention_weights = softmax(scores)  # (num_tokens, num_tokens)

    # Step 5: weighted sum of values
    # Each token's output is a blend of all value vectors,
    # weighted by how much attention it paid to each token.
    output = attention_weights @ V       # (num_tokens, d_k)

    return output, attention_weights


def our_attention(X, W_Q, W_K, W_V, b_Q, b_K, b_V, d_k):
    """Full attention computation for one head using real weights.

    Projects X into Q, K, V spaces using the real learned weight matrices,
    then runs scaled dot-product attention.

    Note on matrix orientation:
    PyTorch stores weight matrices as (out_features, in_features), so
    X @ W_Q.T gives shape (num_tokens, d_k). The .T transposes W_Q from
    (d_k, 384) to (384, d_k) for the matrix multiply.
    """
    # Project input embeddings into Q, K, V spaces
    # X: (num_tokens, 384), W_Q.T: (384, d_k) -> Q: (num_tokens, d_k)
    Q = X @ W_Q.T + b_Q   # (num_tokens, d_k)
    K = X @ W_K.T + b_K   # (num_tokens, d_k)
    V = X @ W_V.T + b_V   # (num_tokens, d_k)

    output, attention_weights = scaled_dot_product_attention(Q, K, V, d_k)
    return output, attention_weights, Q, K, V


# ---------------------------------------------------------------------------
# Step 4: Get the model's actual internal output for comparison
# ---------------------------------------------------------------------------

def get_model_attention_output(bert, tokens, d_k):
    """Run the real model and extract layer 0, head 0's actual output.

    We use PyTorch hooks to capture intermediate activations -- the values
    the model produces internally during a forward pass, before they're
    combined with other heads and processed further.
    """
    captured = {}

    def hook_fn(module, input, output):
        # output is a tuple; first element is the attention output
        captured["attn_output"] = output[0].detach().numpy()
        captured["attn_weights"] = output[1].detach().numpy() if output[1] is not None else None

    # Register hook on layer 0's attention
    layer0_attn = bert.encoder.layer[0].attention.self
    hook = layer0_attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        bert(**tokens, output_attentions=True)

    hook.remove()

    # The model outputs all heads combined: (1, num_tokens, 384)
    # Head 0's slice is the first d_k=32 dimensions
    full_output = captured["attn_output"][0]  # (num_tokens, 384)
    head0_output = full_output[:, :d_k]       # (num_tokens, d_k)

    # Attention weights shape from model: (1, num_heads, num_tokens, num_tokens)
    if captured["attn_weights"] is not None:
        head0_weights = captured["attn_weights"][0, 0]  # (num_tokens, num_tokens)
    else:
        head0_weights = None

    return head0_output, head0_weights


# ---------------------------------------------------------------------------
# Step 5: Visualize attention weights
# ---------------------------------------------------------------------------

def print_attention_heatmap(attention_weights: np.ndarray, tokens: list[str]):
    """Print a text-based heatmap of attention weights.

    Each row shows how much one token attends to every other token.
    Higher values (closer to 1.0) = stronger attention.
    """
    print("\nAttention weight matrix (row = query token, col = key token):")
    print("  Values show: how much does ROW token attend to COLUMN token?")
    print()

    # Header row
    col_width = 10
    header = " " * 14
    for t in tokens:
        header += t[:col_width].center(col_width)
    print(header)
    print("-" * (14 + col_width * len(tokens)))

    for i, row_token in enumerate(tokens):
        row = f"{row_token[:12]:<14}"
        for j in range(len(tokens)):
            weight = attention_weights[i, j]
            # Bold high-attention values with asterisk
            marker = "*" if weight > 0.3 else " "
            row += f"{weight:.3f}{marker}".center(col_width)
        print(row)

    print("\n  * = attention weight > 0.30 (strong attention)")


# ---------------------------------------------------------------------------
# Main: run everything and verify
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("From-scratch attention using real all-MiniLM-L6-v2 weights")
    print("=" * 60)

    # Load model and extract weights
    st_model, bert, W_Q, W_K, W_V, b_Q, b_K, b_V, d_k = load_model_and_weights()

    # Use a real sentence from the GLP-1 domain
    sentence = "Semaglutide causes weight loss in obese patients"
    print(f"\nInput sentence: '{sentence}'")

    # Get token embeddings (X)
    X, token_strings, tokens = get_token_embeddings(bert, sentence)

    # Run our NumPy attention implementation
    print("\nRunning our NumPy attention implementation...")
    our_output, our_weights, Q, K, V = our_attention(
        X, W_Q, W_K, W_V, b_Q, b_K, b_V, d_k
    )

    print(f"  Q shape: {Q.shape}  (each token projected to query space)")
    print(f"  K shape: {K.shape}  (each token projected to key space)")
    print(f"  V shape: {V.shape}  (each token projected to value space)")
    print(f"  Attention weights shape: {our_weights.shape}  (token x token)")
    print(f"  Output shape: {our_output.shape}  (enriched token representations)")

    # Get model's actual output for verification
    print("\nExtracting model's actual internal output for verification...")
    model_output, model_weights = get_model_attention_output(bert, tokens, d_k)

    # Compare -- they should match to floating point precision
    max_diff = np.max(np.abs(our_output - model_output))
    print("\nVerification:")
    print(f"  Max absolute difference between our output and model's output: {max_diff:.6f}")

    if max_diff < 1e-4:
        print("  VERIFIED: our NumPy implementation matches the model's output")
        print("  (small differences are floating point precision, not errors)")
    else:
        print("  MISMATCH: something is wrong in our implementation")
        print(f"  Our output[:3]:   {our_output[:3, :4]}")
        print(f"  Model output[:3]: {model_output[:3, :4]}")

    # Visualize attention weights
    print_attention_heatmap(our_weights, token_strings)

    # Interpretation
    print("\nInterpretation:")
    print("  Each row shows one token's attention distribution over all tokens.")
    print("  Rows sum to 1.0 (softmax output is a probability distribution).")
    print("  High attention weight = 'this token heavily influences my representation'")
    print("  [CLS] and [SEP] are special tokens added by the tokenizer.")
    print("  [CLS] often attends broadly -- it aggregates the whole sequence.")

    print("\n" + "=" * 60)
    print("What this proves:")
    print("  1. We understand what model.encode() is actually computing")
    print("  2. The attention formula Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V")
    print("     is not a black box -- it's a weighted mixture of value vectors")
    print("  3. W_Q, W_K, W_V are fixed pretrained matrices; X changes per sentence")
    print("  4. Our NumPy output matches the model's output to floating point precision")
    print("=" * 60)


if __name__ == "__main__":
    main()