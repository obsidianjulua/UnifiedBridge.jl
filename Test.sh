#!/usr/bin/bash julia

# Load engine
engine = Engine("bert-base-uncased")

# --- Tokenization & Decoding ---
text = "Neural networks are powerful tools."
tokens = encode_text(engine, text)
@show tokens
@show decode_tokens(engine, tokens)

# --- Embeddings & Similarity ---
emb = get_embeddings(engine, text)
@show size(emb)

similarity = text_similarity(engine, "Deep learning", "Neural networks")
@show similarity

# --- Topic Detection ---
result = detect_topics(engine, "Treatment options for patient diagnosis")
@show result

# --- Classification by Examples ---
examples_dict = Dict(
    "positive" => ["This product is amazing and works great!"],
    "negative" => ["Terrible experience, would not recommend."],
    "neutral"  => ["Just average, nothing special."]
)
classified = classify_by_examples(engine, "Absolutely fantastic!", examples_dict)
@show classified

# --- Nearest Neighbor Classification ---
nn_examples = [("positive", "great"), ("negative", "bad"), ("neutral", "meh")]
nearest = nearest_neighbor_classify(engine, "electric car", nn_examples)
@show nearest

# --- Fuzzy Semantic Search ---
docs = [
    "Machine learning enables predictive modeling.",
    "Recurrent networks handle sequence data.",
    "Healthcare systems involve treatment and diagnosis.",
    "Business strategies are key for growth."
]
fuzzy = fuzzy_semantic_search(engine, ["internet"], docs)
@show fuzzy

# --- Embedding Normalization & Dimensionality Reduction ---
norm_emb = normalize_embeddings(Float64.(emb))
@show norm_emb[1:5]

reduced = reduce_dimensions(hcat(Float64.(emb), Float64.(emb)))
@show size(reduced)

# --- Introspection ---

example_usage()
