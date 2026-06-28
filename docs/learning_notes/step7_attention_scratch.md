# Step 7 — From-scratch attention mechanism

## What was built
- `src/scratch/attention.py`: scaled dot-product attention implemented
  in NumPy using real pretrained weights extracted from all-MiniLM-L6-v2
  (the same model used throughout this project for embeddings).
- Verified by comparing NumPy output to the model's actual internal
  activations via PyTorch hooks. Max absolute difference: 0.000000 --
  perfect match to floating point precision.
- Visualized attention weights as a token x token heatmap, showing which
  tokens attend to which in layer 0, head 0.

## Architecture of all-MiniLM-L6-v2 (relevant to this step)
- 6 transformer layers
- 12 attention heads per layer
- Hidden dimension (d_model): 384
- Head dimension (d_k): 384 / 12 = 32
- We extracted layer 0, head 0 -- same math applies to all 72 heads

## The attention formula
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

Step by step:
1. Project input X through learned weight matrices:
   Q = X @ W_Q.T + b_Q  -- "what am I looking for?"
   K = X @ W_K.T + b_K  -- "what do I contain?"
   V = X @ W_V.T + b_V  -- "what should I share?"

2. Compute raw attention scores:
   scores = Q @ K.T      -- (num_tokens, num_tokens)
   scores[i][j] = "how much should token i attend to token j?"

3. Scale to prevent softmax saturation:
   scores = scores / sqrt(d_k)
   Without this, large d_k causes large dot products, pushing softmax
   toward 0/1 extremes and making gradients vanish during training.

4. Softmax -- turn scores into weights summing to 1:
   weights = softmax(scores)   -- (num_tokens, num_tokens)

5. Weighted sum of values:
   output = weights @ V        -- (num_tokens, d_k)
   Each token's output is a blend of all value vectors, weighted by
   how much attention it paid to each token.

## X vs. W_Q/K/V -- the critical distinction
- X (token embeddings): input-dependent. Changes completely with every
  new sentence. Shape (num_tokens, 384). Produced by a lookup table
  (embedding matrix) indexed by token IDs, plus positional encoding.
  "What is this specific token?"
- W_Q, W_K, W_V (weight matrices): fixed after training. Same matrices
  for every sentence ever encoded. Shape (32, 384) per head.
  "How do we transform any token's identity into Q/K/V space?"
  Both are learned during pre-training, but serve completely different
  roles: X is a lookup, W is a reusable transformation.

## What the attention weights revealed
Input: "Semaglutide causes weight loss in obese patients"
Tokens: [CLS] se ##ma ##gl ##uti ##de causes weight loss in obe ##se patients [SEP]

Notable patterns (layer 0, head 0):
- "semaglutide" subword tokens (se, ##ma, ##gl, ##uti, ##de) attend
  heavily to [CLS], not strongly to each other -- layer 0 is still
  learning to aggregate subword fragments; deeper layers would show
  stronger intra-word attention
- "causes" attends to "patients" (0.138) and "obese/##se" (0.171) --
  picking up "causes [effect] in [whom]" semantic frame
- "weight" attends to "patients" (0.167) and "causes" (0.127) --
  building clinical context
- "loss" attends to "obe/##se" (0.202) -- associating weight loss
  with obesity context
- [CLS] attends heavily to [SEP] (0.338) -- known BERT behavior where
  [CLS] and [SEP] form structural anchors in early layers

These are real linguistic relationships emerging from learned weights
and the attention formula -- not engineered heuristics.

## Tokenization observation
"semaglutide" → ['se', '##ma', '##gl', '##uti', '##de'] (5 subword tokens)
WordPiece tokenization splits rare/unknown words into known subword pieces.
Drug names are often subword-tokenized because they weren't common in the
pre-training corpus. Relevant for medical RAG: rare drug names get
fragmented, potentially affecting embedding quality compared to common words.
The ## prefix indicates a continuation subword (not a word-initial token).

## Implementation details worth knowing
- Numerically stable softmax: subtract max before exp() to prevent
  overflow. Mathematically identical result, numerically safe.
  exp(x-max)/sum(exp(x-max)) == exp(x)/sum(exp(x))
- PyTorch weight convention: W stored as (out_features, in_features),
  so we use X @ W.T (not X @ W) to get (num_tokens, d_k) output
- BatchSpanProcessor vs per-span: PyTorch hooks used to extract
  intermediate activations without modifying model source code --
  same non-invasive instrumentation philosophy as Phoenix auto-instrumentation
- sdpa attention warning: newer PyTorch uses scaled_dot_product_attention
  (sdpa) by default which doesn't support output_attentions=True --
  harmless warning, our hook captures the output regardless

## What this proves for an interview
1. The formula is not a black box -- implementing it in NumPy and
   verifying against the real model proves understanding of the math
2. The X vs W distinction is concrete -- X shape changes per sentence,
   W shapes are always (32, 384) regardless of input
3. Attention captures real linguistic relationships -- even at layer 0
   head 0, attention weights show "causes" attending to "obese patients"
4. Max absolute difference 0.000000 -- our math is exactly right,
   not approximately right

## Likely interview questions tied to this step
- "Explain attention in transformers from scratch." -> walk through
  Q/K/V projection, QK^T dot product, scaling, softmax, weighted sum
  of V. Use the "what am I looking for / what do I contain / what do
  I share" framing for Q/K/V roles.
- "Why divide by sqrt(d_k)?" -> prevents large dot products from
  saturating softmax, which would make gradients vanish during training.
  As d_k increases, expected dot product magnitude grows as sqrt(d_k),
  so we divide by sqrt(d_k) to normalize back to unit scale.
- "What's the difference between X and the weight matrices W_Q/K/V?"
  -> X is input-dependent (changes per sentence, a lookup), W is
  fixed after training (same for every sentence, a learned transformation).
  Both are learned during pre-training but serve different roles.
- "What is multi-head attention and why use multiple heads?" -> run
  the attention computation in parallel with different W_Q/K/V matrices
  (different "heads"), each learning to capture different types of
  relationships (syntactic, semantic, positional). Outputs concatenated
  and projected back. 12 heads in MiniLM, each with d_k=32, concatenated
  to 384 = full d_model. Different heads specialize automatically.
- "How does self-attention differ from cross-attention?" -> self-attention:
  Q, K, V all come from the same sequence (a token attending to other
  tokens in the same sentence). Cross-attention: Q comes from one
  sequence, K and V from another (used in encoder-decoder architectures
  like translation, where the decoder attends to the encoder's output).
  Our implementation is self-attention.