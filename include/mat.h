#pragma once
#include <stdlib.h>

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

Matrix *mat_create(size_t rows, size_t cols);
void mat_delete(Matrix *m);
void mat_randomize(Matrix *m);
void mat_set(Matrix *m, float n);
void mat_add_scalar(Matrix *a, float n);
void mat_sub_scalar(Matrix *a, float n);
void mat_mul_scalar(Matrix *a, float n);
int mat_add(Matrix *a, Matrix *b);
int mat_sub(Matrix *a, Matrix *b);
int mat_mul_entrywise(Matrix *a, Matrix *b);
Matrix *mat_mul(Matrix *a, Matrix *b, Matrix *out);
Matrix *mat_transpose(Matrix *m);
Matrix *mat_from_array(float *a, size_t size, Matrix *out);

