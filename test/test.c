#include "nn.h"
#include <stdio.h>

int main(void) {
    size_t shape[] = { 2, 4, 1, 0 };
    NeuralNetwork *nn = nn_create(shape);
    puts("Created a new NeuralNetwork:");
    printf("depth = %lu\n", nn->depth);
    puts("shape = {");
    for (size_t i = 0; i < nn->depth; i++) {
        printf("  %lu: %lu\n",
                i, nn->shape[i]);
    }
    puts("}\nlayers = {");
    for (size_t i = 0; i < nn->depth; i++) {
        printf("  %lu: [%lu, %lu]\n",
                i, nn->layers[i]->rows, nn->layers[i]->cols);
    }
    puts("}\nweights = {");
    for (size_t i = 0; i < nn->depth - 1; i++) {
        printf("  %lu: [%lu, %lu]\n",
                i, nn->weights[i]->rows, nn->weights[i]->cols);
    }
    puts("}\nbiases = {");
    for (size_t i = 0; i < nn->depth - 1; i++) {
        printf("  %lu: [%lu, %lu]\n",
                i, nn->biases[i]->rows, nn->biases[i]->cols);
    }
    puts("}");

    nn_delete(nn);
    puts("Deleted the NeuralNetwork");

    return 0;
}
