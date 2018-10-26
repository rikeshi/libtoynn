//#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DTYPE float
#define NN(shape) nn_create(shape, sizeof(shape)/sizeof(shape[0]))


typedef struct NeuralNetwork {
  size_t memsize;
  size_t depth;
  DTYPE **layers;
  DTYPE **weights;
  DTYPE **biases;
  size_t *shape;
} NeuralNetwork;


/*
 * Memory layout:
 * +-------------------------------------------------------+
 * | 0. struct NeuralNetwork                               |
 * |    - depth                                            |
 * |    - *layers[]                                        |
 * |    - *weights[]                                       |
 * |    - *biases[]                                        |
 * |    - *shape[]                                         |
 * +-------------------------------------------------------+
 * | 1. layers[]: layers[i] -> layer_arrays[i]             |
 * | 2. weights[]: weights[i] -> weight_arrays[i]          |
 * | 3. biases[]: biases[i] -> bias_arrays[i]              |
 * | 4. lengths[]: lengths[i] -> length of layer_arrays[i] |
 * +-------------------------------------------------------+
 * | 5. layer_arrays[]                                     |
 * | 6. weight_arrays[]                                    |
 * | 7. bias_arrays[]                                      |
 * +-------------------------------------------------------+
 *
 * store neural network in a single malloc block for easy cloning etc.
 */
NeuralNetwork *nn_create(size_t *shape, size_t depth) {
  size_t size = sizeof(NeuralNetwork); // struct
  size += (depth * 3 - 2) * sizeof(DTYPE*); // pointer arrays
  size += depth * sizeof(size_t); // shape
  size += shape[0] * sizeof(DTYPE); // first layer
  for (size_t i = 1; i < depth; i++) { // other layers, weights, biases
    size += (shape[i-1] + 2) * shape[i] * sizeof(DTYPE);
  }

  NeuralNetwork *nn = malloc(size);
  nn->memsize = size;
  nn->depth = depth;
  nn->layers = (DTYPE**)((char*)nn + sizeof(NeuralNetwork));
  nn->weights = nn->layers + depth;
  nn->biases = nn->weights + depth-1;
  nn->shape = (size_t*)(nn->biases + depth-1);

  // set shape and pointer arrays
  memcpy(nn->shape, shape, depth * sizeof(size_t));
  nn->layers[0] = (DTYPE*)(nn->shape + depth);
  for (size_t i = 1; i < depth; i++) {
    nn->layers[i] = nn->layers[i-1] + shape[i-1];
  }
  nn->weights[0] = nn->layers[depth-1] + shape[depth-1];
  for (size_t i = 1; i < depth-1; i++) {
    nn->weights[i] = nn->weights[i-1] + (shape[i-1] * shape[i]);
  }
  nn->biases[0] = nn->weights[depth-2] + shape[depth-1];
  for (size_t i = 1; i < depth-1; i++) {
    nn->biases[i] = nn->biases[i-1] + shape[i];
  }

  return nn;
}


void nn_delete(NeuralNetwork *nn) {
  free(nn);
}


void nn_print(NeuralNetwork *nn) {
  printf("-- Neural Network --\n");
  printf("memsize: %lu bytes\n", nn->memsize);
  printf("depth: %lu layers\n", nn->depth);
  printf("shape: [ ");
  for (size_t i = 0; i < nn->depth; i++) {
    printf("%lu ", nn->shape[i]);
  }
  printf("]");
  for (size_t i = 0; i < nn->depth; i++) {
    printf("\nlayer %lu:\n", i);
    for (int j = 0; j < nn->shape[i]; j++) {
      printf("%f, ", nn->layers[i][j]);
    }
  }
  printf("\n");
  for (size_t i = 1; i < nn->depth; i++) {
    printf("\nweights %lu:\n", i);
    for (int j = 0; j < nn->shape[i-1] * nn->shape[i]; j++) {
      printf("%f, ", nn->weights[i-1][j]);
    }
  }
  printf("\n");
  for (size_t i = 1; i < nn->depth; i++) {
    printf("\nbiases %lu:\n", i);
    for (int j = 0; j < nn->shape[i]; j++) {
      printf("%f, ", nn->biases[i-1][j]);
    }
  }
  printf("\n");
}


void nn_print_bytes(NeuralNetwork *nn) {
  size_t i = 0;
  unsigned char *ptr = (char*)nn;
  while(ptr < (unsigned char*)nn + nn->memsize) {
    printf("%u ", *ptr++);
    if (++i == 8) {
      printf("\n");
      i = 0;
    }
  }
  printf("\n");
}

NeuralNetwork *nn_alloc(size_t *shape, size_t depth) {
  size_t size = sizeof(NeuralNetwork); // struct
  size += (depth * 3 - 2) * sizeof(DTYPE*); // pointer arrays
  size += depth * sizeof(size_t); // shape
  size += shape[0] * sizeof(DTYPE); // first layer
  for (size_t i = 1; i < depth; i++) { // layers, weights, biases
    size += (shape[i-1] + 2) * shape[i] * sizeof(DTYPE);
  }
  return malloc(size);
}


int main(void) {
  size_t shape[] = { 2, 3, 1 };
  NeuralNetwork *nn = NN(shape);
  NeuralNetwork *nn2 = nn_alloc(shape, 3);
  nn_print(nn);
  puts("");
  nn_print_bytes(nn); // there are some gaps of 4 bits in the pointer arrays
  nn_delete(nn);

  return 0;
}
