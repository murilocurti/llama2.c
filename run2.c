#include <stdio.h>

#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "llama.h"

int main()
{
    printf("Hello, World!\n");

    GeneratorParams *params = malloc(sizeof(GeneratorParams));

    params->checkpoint = "C:\\Users\\MuriloCurti\\source\\repos\\.murilocurti\\stories15M.bin";
    // params->checkpoint = "C:\\Users\\MuriloCurti\\source\\repos\\.murilocurti\\llama2.c\\llama2_7b.bin";
    params->steps = 256;
    params->temperature = 0.0f;
    GeneratorContext *context = initialize(params);

    if (context == NULL)
    {
        printf("Initialization failed. Exiting.\n");
        return -1;
    }

    printf("Initialization successful.\n");
    // Generate and print tokens.
    int num_tokens = params->steps; // Adjust this to the desired number of tokens.

    printf("Generating %d tokens...\n", num_tokens);

    for (int i = 0; i < num_tokens; i++)
    {
        char *token = next_token(context);
        printf("%s", token);
        fflush(stdout);
    }

    printf("generation complete.\n");

    // Report stats and cleanup.
    report_stats(context);
    cleanup(context);

    return 0;
}