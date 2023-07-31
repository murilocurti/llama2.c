/*
Inference for Llama-2 Transformer model in pure C.

Example compile: (see README for more details)
$ gcc -O3 -o run run.c -lm

Then run with:
$ ./run
*/

#include <stdio.h>

#include <math.h>
#include <string.h>
#include <fcntl.h>

#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "transformer.h"
#include "llama.h"

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
            return NULL;
        }

        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1)
        {
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
            return NULL;
        }

        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

        if (data == MAP_FAILED)
        {
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
    int next;

    Config config = context->config;
    RunState state = context->state;
    TransformerWeights weights = context->weights;
    char **vocab = context->vocab;
    float temperature = context->temperature;
    int token = context->token;
    int pos = context->pos;

    transformer(token, pos, &config, &state, &weights);

    if (temperature == 0.0f)
    {
        next = argmax(state.logits, config.vocab_size);
    }
    else
    {
        for (int q = 0; q < config.vocab_size; q++)
        {
            state.logits[q] /= temperature;
        }

        softmax(state.logits, config.vocab_size);
        next = sample(state.logits, config.vocab_size);
    }

    char *token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next] + 1 : vocab[next];

    token = next;
    pos++;
    context->token = token;
    context->pos = pos;

    if (context->start == 0)
    {
        context->start = time_in_ms();
    }

    return token_str;
}

int report_stats(GeneratorContext *context)
{
    long end = time_in_ms();
    return (context->steps - 1) / (double)(end - context->start) * 1000;
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