#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "structs.h"

void malloc_run_state(RunState *s, Config *p);
void free_run_state(RunState *s);
void checkpoint_init_weights(TransformerWeights *w, Config *p, float *f, int shared_weights);
void accum(float *a, float *b, int size);
void rmsnorm(float *o, float *x, float *weight, int size);
void softmax(float *x, int size);
void matmul(float *xout, float *x, float *w, int n, int d);
void transformer(int token, int pos, Config *p, RunState *s, TransformerWeights *w);
int sample(float *probabilities, int n);
int argmax(float *v, int n);
long time_in_ms();

#endif