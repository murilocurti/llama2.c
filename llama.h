#ifndef LLAMA_H
#define LLAMA_H

#include "structs.h"

// Declare the functions exposed by your library.
GeneratorContext *initialize(GeneratorParams *params);
char *next_token(GeneratorContext *context);
int report_stats(GeneratorContext *context);
void cleanup(GeneratorContext *context);

#endif
