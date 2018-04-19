#pragma once
#include "mat.h"
#include <math.h>
#include <stdlib.h>

typedef struct NeuralNetwork {
    size_t depth;
    size_t *shape;
    Matrix **layers;
    Matrix **weights;
    Matrix **biases;
    void (*activation)(Matrix *m);
    void (*dactivation)(Matrix *m);
    int (*dcost)(Matrix *target, Matrix *m);
    float lr;
} NeuralNetwork;

void nn_sigmoid(Matrix *m);
void nn_dsigmoid(Matrix *m);
void nn_tanh(Matrix *m);
void nn_dtanh(Matrix *m);
void nn_relu(Matrix *m);
void nn_drelu(Matrix *m);
int nn_mse_gradient(Matrix *y, Matrix *m);
NeuralNetwork *nn_create(size_t shape[]);
void nn_delete(NeuralNetwork *nn);
Matrix *nn_predict(NeuralNetwork *nn, float *x);
void nn_train(NeuralNetwork *nn, float *y);

