/*
Inference for Llama-2 Transformer model in pure C.

Example compile: (see README for more details)
$ gcc -O3 -o run run.c -lm
$ gcc -shared -o lib.so lib.c -lm

Then run with:
$ ./run
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "lib.h"
// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

void malloc_run_state(RunState *s, Config *p)
{
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache)
    {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState *s)
{
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

void checkpoint_init_weights(TransformerWeights *w, Config *p, float *f, int shared_weights)
{
    float *ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wk = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wv = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->freq_cis_real = ptr;
    int head_size = p->dim / p->n_heads;
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(float *a, float *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[i] += b[i];
    }
}

void rmsnorm(float *o, float *x, float *weight, int size)
{
    // calculate sum of squares
    float ss = 0.0f;

    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }

    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    // normalize and scale
    for (int j = 0; j < size; j++)
    {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void matmul(float *xout, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;

#pragma omp parallel for private(i)

    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void transformer(int token, int pos, Config *p, RunState *s, TransformerWeights *w)
{
    // printf("debug tranformer call\n");
    // printf("token: %d\n", token);
    // printf("pos: %d\n", pos);
    // printf("config address %p\n", p);
    // printf("runstate address %p\n", s);
    // printf("weights address %p\n", w);

    // a few convenience variables
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float *content_row = &(w->token_embedding_table[token * dim]);

    // printf("lets2.1");
    // // Print the values of dim and pointers before memcpy
    // printf("Before memcpy:\n");
    // printf("dim: %d\n", dim);
    // printf("x pointer: %p\n", (void *)x);
    // printf("content_row pointer: %p\n", (void *)content_row);

    // Print some elements of content_row to verify its contents
    // printf("content_row contents:\n");
    // for (int i = 0; i < dim; i++)
    // {
    //     printf("content_row[%d]: %d\n", i, content_row[i]);
    // }

    // return;
    memcpy(x, content_row, dim * sizeof(*x));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float *freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float *freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++)
    {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p->n_heads; h++)
        {
            // get the q and k vectors for this head
            float *q = s->q + h * head_size;
            float *k = s->k + h * head_size;
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i += 2)
            {
                float q0 = q[i];
                float q1 = q[i + 1];
                float k0 = k[i];
                float k1 = k[i + 1];
                float fcr = freq_cis_real_row[i / 2];
                float fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float *key_cache_row = s->key_cache + loff + pos * dim;
        float *value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++)
        {
            // get the query vector for this head
            float *q = s->q + h * head_size;
            // attention scores for this head
            float *att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++)
            {
                // get the key vector for this head and at this timestep
                float *k = s->key_cache + loff + t * dim + h * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
                // get the value vector for this head and at this timestep
                float *v = s->value_cache + loff + t * dim + h * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++)
        {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }

        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++)
        {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

int sample(float *probabilities, int n)
{
    // sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;

    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        if (r < cdf)
        {
            return i;
        }
    }

    return n - 1; // in case of rounding errors
}

int argmax(float *v, int n)
{
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];

    for (int i = 1; i < n; i++)
    {
        if (v[i] > max_p)
        {
            max_i = i;
            max_p = v[i];
        }
    }

    return max_i;
}

// ----------------------------------------------------------------------------

long time_in_ms()
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

GeneratorContext *initialize(GeneratorParams *params)
{
    GeneratorContext *context = malloc(sizeof(GeneratorContext));

    // Copy your initialization code here
    context->temperature = params->temperature >= 0 ? params->temperature : 0.9;
    context->steps = params->steps > 0 ? params->steps : 256;

    srand((unsigned int)time(NULL));

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int fd = 0;         // file descriptor for memory mapping
    float *data = NULL; // memory mapped data pointer
    long file_size;     // size of the checkpoint file in bytes

    {
        FILE *file = fopen(params->checkpoint, "rb");

        if (!file)
        {
            printf("Couldn't open file %s\n", params->checkpoint);

            return NULL;
        }

        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1)
        {
            printf("Couldn't read config\n");

            return NULL;
        }

        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file);  // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        fd = open(params->checkpoint, O_RDONLY); // open in read only mode

        if (fd == -1)
        {
            printf("open failed!\n");

            return NULL;
        }

        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

        if (data == MAP_FAILED)
        {
            printf("mmap failed!\n");

            return NULL;
        }

        float *weights_ptr = data + sizeof(Config) / sizeof(float);
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
    }

    char **vocab = (char **)malloc(config.vocab_size * sizeof(char *));
    {
        FILE *file = fopen("tokenizer.bin", "rb");
        if (!file)
        {
            printf("Couldn't load tokenizer.bin\n");
            return NULL;
        }
        int len;
        for (int i = 0; i < config.vocab_size; i++)
        {
            if (fread(&len, sizeof(int), 1, file) != 1)
            {
                return NULL;
            }
            vocab[i] = (char *)malloc(len + 1);
            if (fread(vocab[i], len, 1, file) != 1)
            {
                return NULL;
            }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

    RunState state;

    malloc_run_state(&state, &config);

    context->state = state;
    context->config = config;
    context->weights = weights;
    context->vocab = vocab;

    context->token = 1;
    context->pos = 0;
    context->start = 0;

    return context;
}

char *next_token(GeneratorContext *context)
{
    // printf("\rGenerating token 1 %d/%d\n", context->pos, context->steps);
    int next;

    // printf("debug transfromer call");
    // printf("token: %d\n", context->token);
    // printf("pos: %d\n", context->pos);
    // printf("config: %d\n", context->config);
    // printf("state: %d\n", context->state);
    // printf("weights: %d\n", context->weights);

    // printf("debug tranformer call\n");
    // printf("token: %d\n", context->token);
    // printf("pos: %d\n", context->pos);
    // printf("config.dim: %d\n", context->config.dim);
    // printf("config.hidden_dim: %d\n", context->config.hidden_dim);
    // printf("config.n_layers: %d\n", context->config.n_layers);
    // printf("config.n_heads: %d\n", context->config.n_heads);
    // printf("config.n_kv_heads: %d\n", context->config.n_kv_heads);
    // printf("config.vocab_size: %d\n", context->config.vocab_size);
    // printf("config.seq_len: %d\n", context->config.seq_len);
    // printf("state.x: %f\n", context->state.x);
    // printf("state.xb: %f\n", context->state.xb);
    // printf("state.xb2: %f\n", context->state.xb2);
    // printf("state.hb: %f\n", context->state.hb);
    // printf("state.hb2: %f\n", context->state.hb2);
    // printf("state.q: %f\n", context->state.q);
    // printf("state.k: %f\n", context->state.k);
    // printf("state.v: %f\n", context->state.v);
    // printf("state.att: %f\n", context->state.att);
    // printf("state.logits: %f\n", context->state.logits);
    // printf("state.key_cache: %f\n", context->state.key_cache);
    // printf("state.value_cache: %f\n", context->state.value_cache);
    // printf("weights.token_embedding_table: %f\n", context->weights.token_embedding_table);
    // printf("weights.rms_att_weight: %f\n", context->weights.rms_att_weight);
    // printf("weights.wq: %f\n", context->weights.wq);
    // printf("weights.wk: %f\n", context->weights.wk);
    // printf("weights.wv: %f\n", context->weights.wv);
    // printf("weights.wo: %f\n", context->weights.wo);
    // printf("weights.rms_ffn_weight: %f\n", context->weights.rms_ffn_weight);
    // printf("weights.w1: %f\n", context->weights.w1);
    // printf("weights.w2: %f\n", context->weights.w2);
    // printf("weights.w3: %f\n", context->weights.w3);
    // printf("weights.rms_final_weight: %f\n", context->weights.rms_final_weight);
    // printf("weights.freq_cis_real: %f\n", context->weights.freq_cis_real);
    // printf("weights.freq_cis_imag: %f\n", context->weights.freq_cis_imag);
    // printf("weights.wcls: %f\n", context->weights.wcls);

    Config config = context->config;
    RunState state = context->state;
    TransformerWeights weights = context->weights;
    char **vocab = context->vocab;
    float temperature = context->temperature;
    int token = context->token;
    int pos = context->pos;

    // printf("debug tranformer call\n");
    // printf("token: %d\n", context->token);
    // printf("pos: %d\n", context->pos);
    // printf("config address %p\n", config);
    // printf("runstate address %p\n", state);
    // printf("weights address %p\n", weights);

    transformer(token, pos, &config, &state, &weights);
    // transformer(1, 0, NULL, NULL, NULL);

    // printf("\rGenerating token 2 %d/%d", context->pos, context->steps);
    // prsintf("%d", temperature);
    if (temperature == 0.0f)
    {
        // printf("\rGenerating token 3 %d/%d", context->pos, context->steps);
        next = argmax(state.logits, config.vocab_size);
    }
    else
    {
        // printf("\rGenerating token 4 %d/%d", context->pos, context->steps);

        for (int q = 0; q < config.vocab_size; q++)
        {
            state.logits[q] /= temperature;
        }

        softmax(state.logits, config.vocab_size);
        next = sample(state.logits, config.vocab_size);
    }

    // printf("\rGenerating token 5 %d/%d", context->pos, context->steps);

    // printf("token: %d\n", context->token);
    // printf("next: %d\n", next);
    // printf("context->token: %d\n", context->token);
    // printf("context->config.vocab_size: %d\n", context->config.vocab_size);
    // printf("context->vocab[next][0]: %c\n", context->vocab[next][0]);
    // printf("context->vocab[next][0]: %c\n", context->vocab[next][0]);
    // printf("context->vocab[next]: %s\n", context->vocab[next]);

    char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next] + 1 : vocab[next];

    // printf("\rGenerating token 5.1 %d/%d", context->pos, context->steps);
    token = next;
    pos++;
    context->token = token;
    context->pos = pos;

    if (context->start == 0)
    {
        context->start = time_in_ms();
    }

    // printf("\rGenerating token 6 %d/%d", context->pos, context->steps);
    return token_str;
}

void report_stats(GeneratorContext *context)
{
    long end = time_in_ms();
    printf("\nachieved tok/s: %f\n", (context->steps - 1) / (double)(end - context->start) * 1000);
}

void cleanup(GeneratorContext *context)
{
    free_run_state(&context->state);

    for (int i = 0; i < context->config.vocab_size; i++)
    {
        free(context->vocab[i]);
    }

    free(context->vocab);

    if (context->data != MAP_FAILED)
    {
        munmap(context->data, context->file_size);
    }

    if (context->fd != -1)
    {
        close(context->fd);
    }

    free(context);
}