#include "matrix.h"

Matrix *create_matrix(size_t rows, size_t cols) {
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols * sizeof(float));
    return m;
}

int destroy_matrix(Matrix *m) {
    if (!m) return 1;
    free(m->data);
    free(m);
    return 0;
}

Matrix *fill_matrix(Matrix *m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        m->data[i] = n;
    return m;
}

Matrix *add_scalar(Matrix* m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        m->data[i] = m->data[i] + n;
    return m;
}

Matrix *sub_scalar(Matrix* m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        m->data[i] = m->data[i] - n;
    return m;
}

Matrix *mult_scalar(Matrix *m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++)
        m->data[i] = m->data[i] * n;
    return m;
}

Matrix *add_elem(Matrix *m1, Matrix *m2) {
    if (m1->rows != m2->rows) return NULL;
    if (m1->cols != m2->cols) return NULL;
    for (size_t i = 0; i < m1->rows * m1->cols; i++)
        m1->data[i] = m1->data[i] + m2->data[i];
    return m1;
}

Matrix *sub_elem(Matrix *m1, Matrix *m2) {
    if (m1->rows != m2->rows) return NULL;
    if (m1->cols != m2->cols) return NULL;
    for (size_t i = 0; i < m1->rows * m1->cols; i++)
        m1->data[i] = m1->data[i] - m2->data[i];
    return m1;
}

Matrix *mult_elem(Matrix *m1, Matrix *m2) {
    if (m1->rows != m2->rows) return NULL;
    if (m1->cols != m2->cols) return NULL;
    for (size_t i = 0; i < m1->rows * m1->cols; i++)
        m1->data[i] = m1->data[i] * m2->data[i];
    return m1;
}
Matrix *mult_mat(Matrix *m1, Matrix *m2, Matrix *mout) {
    if (m1->cols != m2->rows) return NULL;
    if (!mout) mout = create_matrix(m1->rows, m2->cols);
    for (size_t i = 0; i < mout->rows; i++) {
        for (size_t j = 0; j < mout->cols; j++) {
            float dot_product = 0;
            for(size_t k = 0; k < m1->cols; k++) {
                float x1 = m1->data[i * m1->cols + k];
                float x2 = m2->data[k * m2->cols + j];
                dot_product += x1 * x2;
            }
            mout->data[i * mout->cols + j] = dot_product;
        }
    }
    return mout;
}

Matrix *transpose(Matrix *m, Matrix *mout) {
    if (!mout) mout = create_matrix(m->cols, m->rows);
    for (size_t i = 0; i < m->rows; i++)
        for (size_t j = 0; j < m->cols; j++)
            mout->data[j * m->rows + i] = m->data[i * m->cols + j];
    return mout;
}

Matrix *randomize(Matrix *m) {
    // set all values randomly between -1 and 1
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        float r = rand() / (float)RAND_MAX;
        m->data[i] = r * 2 - 1;
    }
    return m;
}

Matrix *matrix_from_array(float *a, size_t size, Matrix *mout) {
    if (!mout) mout = create_matrix(size, 1);
    for (size_t i = 0; i < size; i++)
        mout->data[i] = a[i];
    return mout;
}

void print_matrix(Matrix *m) {
    printf("shape = [%u, %u]\n", m->rows, m->cols);
    puts("{");
    for (size_t i = 0; i < m->rows; i++) {
        printf("  ");
        for (size_t j = 0; j < m->cols; j++)
            printf("%1.4f ", m->data[i * m->cols + j]);
        puts("");
    }
    puts("}");
}
