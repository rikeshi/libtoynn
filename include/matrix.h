#pragma once
#include <stdio.h>
#include <stdlib.h>

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

Matrix *create_matrix(size_t rows, size_t cols);
int destroy_matrix(Matrix *m);
Matrix *fill_matrix(Matrix *m, float n);
Matrix *add_scalar(Matrix *m, float n);
Matrix *sub_scalar(Matrix *m, float n);
Matrix *mult_scalar(Matrix *m, float n);
Matrix *add_elem(Matrix *m1, Matrix *m2);
Matrix *sub_elem(Matrix *m1, Matrix *m2);
Matrix *mult_elem(Matrix *m1, Matrix *m2);
Matrix *mult_mat(Matrix *m1, Matrix *m2, Matrix *mout);
Matrix *transpose(Matrix *m, Matrix *mout);
Matrix *randomize(Matrix *m);
Matrix *matrix_from_array(float *a, size_t size, Matrix *mout);
void print_matrix(Matrix *m);

