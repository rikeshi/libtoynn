#include "nn.h"

void nn_sigmoid(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = 1.0f / (1.0f + exp(-x));
    }
}

void nn_dsigmoid(Matrix *m) {
    for(size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = x * (1.0f - x);
    }
}

void nn_tanh(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = tanhf(x);
    }
}

void nn_dtanh(Matrix *m) {
    for(size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = 1.0f - (x * x);
    }
}

void nn_relu(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = fmaxf(0.0f, x);
    }
}

void nn_drelu(Matrix *m) {
    for(size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = (x > 0.0f) ? x : 0.0f;
    }
}

int nn_mse_gradient(Matrix *y, Matrix *m) {
    if (y->rows != m->rows) return 1;
    if (y->cols != m->cols) return 1;
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        y->data[i] = y->data[i] - m->data[i];
    }
    return 0;
}

NeuralNetwork *nn_create(size_t shape[]) {
    NeuralNetwork *nn = malloc(sizeof (NeuralNetwork));
    size_t *end = shape;
    while (*end) end++;
    nn->depth = end - shape;
    nn->shape = malloc(nn->depth * (sizeof *nn->shape));
    nn->layers = malloc(nn->depth * (sizeof *nn->layers));
    nn->weights = malloc((nn->depth - 1) * (sizeof *nn->weights));
    nn->biases = malloc((nn->depth - 1) * (sizeof *nn->biases));
    for (size_t i = 0; i < nn->depth; i++) {
        nn->shape[i] = shape[i];
        nn->layers[i] = mat_create(shape[i], 1);
        if (i != nn->depth - 1) {
            nn->weights[i] = mat_create(shape[i+1], shape[i]);
            nn->biases[i] = mat_create(shape[i+1], 1);
            mat_randomize(nn->weights[i]);
            mat_randomize(nn->biases[i]);
        }
    }
    nn->activation = nn_sigmoid;
    nn->dactivation = nn_dsigmoid;
    nn->dcost = nn_mse_gradient;
    nn->lr = 0.1;
    return nn;
}

void nn_delete(NeuralNetwork *nn) {
    free(nn->shape);
    for (size_t i = 1; i < nn->depth - 1; i++) {
        mat_delete(nn->layers[i]);
        mat_delete(nn->weights[i]);
        mat_delete(nn->biases[i]);
    }
    mat_delete(nn->layers[nn->depth - 1]);
    free(nn);
}

Matrix *nn_predict(NeuralNetwork *nn, float *x) {
    mat_from_array(x, nn->shape[0], nn->layers[0]);
    for (size_t i = 1; i < nn->depth; i++) {
        mat_mul(nn->weights[i-1], nn->layers[i-1], nn->layers[i]);
        mat_add(nn->layers[i], nn->biases[i-1]);
        nn->activation(nn->layers[i]);
    }
    return nn->layers[nn->depth - 1];
}

void nn_train(NeuralNetwork *nn, float *y) {
    Matrix *err = mat_from_array(y, nn->shape[nn->depth - 1], NULL);
    for (size_t i = nn->depth - 1; i > 0; i--) {
        if (i == nn->depth - 1) {
            nn->dcost(err, nn->layers[nn->depth - 1]);
            nn->dactivation(nn->layers[nn->depth - 1]);
        } else {
            Matrix *weights_t = mat_transpose(nn->weights[i]);
            Matrix *prev_err = err;
            err = mat_mul(weights_t, prev_err, NULL);
            mat_delete(weights_t);
            mat_delete(prev_err);
        }
        nn->dactivation(nn->layers[i]);
        mat_mul_entrywise(nn->layers[i], err);
        mat_mul_scalar(nn->layers[i], nn->lr);
        Matrix *layer_t = mat_transpose(nn->layers[i-1]);
        Matrix *deltas = mat_mul(nn->layers[i], layer_t, NULL);
        mat_add(nn->weights[i-1], deltas);
        mat_add(nn->biases[i-1], nn->layers[i]);
        mat_delete(layer_t);
        mat_delete(deltas);
    }
}

