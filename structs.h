#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct
{
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
} Config;

typedef struct
{
    // token embedding table
    float *token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float *wq; // (layer, dim, dim)
    float *wk; // (layer, dim, dim)
    float *wv; // (layer, dim, dim)
    float *wo; // (layer, dim, dim)
    // weights for ffn
    float *w1; // (layer, hidden_dim, dim)
    float *w2; // (layer, dim, hidden_dim)
    float *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float *freq_cis_real; // (seq_len, dim/2)
    float *freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float *wcls;
} TransformerWeights;

typedef struct
{
    // current wave of activations
    float *x;      // activation at current time stamp (dim,)
    float *xb;     // same, but inside a residual branch (dim,)
    float *xb2;    // an additional buffer just for convenience (dim,)
    float *hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;      // query (dim,)
    float *k;      // key (dim,)
    float *v;      // value (dim,)
    float *att;    // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct
{
    Config config;
    TransformerWeights weights;
    RunState state;
    float *data;
    int fd;
    char **vocab;
    float temperature;
    int steps;
    int pos;
    int token;
    long start;
    size_t file_size;
} GeneratorContext;

typedef struct GeneratorParams
{
    char *checkpoint;
    float temperature;
    int steps;
} GeneratorParams;

#endif