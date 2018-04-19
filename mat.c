#include "mat.h"

Matrix *mat_create(size_t rows, size_t cols) {
    Matrix *m = malloc(sizeof (Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * cols * (sizeof *m->data));
    return m;
}

void mat_delete(Matrix *m) {
    free(m->data);
    free(m);
}

void mat_randomize(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        m->data[i] = r * 2.0f - 1.0f;
    }
}

void mat_fill(Matrix *m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = n;
    }
}

void mat_add_scalar(Matrix *m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = m->data[i] + n;
    }
}

void mat_sub_scalar(Matrix *m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = m->data[i] - n;
    }
}

void mat_mul_scalar(Matrix *m, float n) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = m->data[i] * n;
    }
}

int mat_add(Matrix *a, Matrix *b) {
    if (a->rows != b->rows) return 1;
    if (a->cols != b->cols) return 1;
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        a->data[i] = a->data[i] + b->data[i];
    }
    return 0;
}

int mat_sub(Matrix *a, Matrix *b) {
    if (a->rows != b->rows) return 1;
    if (a->cols != b->cols) return 1;
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        a->data[i] = a->data[i] - b->data[i];
    }
    return 0;
}

int mat_mul_entrywise(Matrix *a, Matrix *b) {
    if (a->rows != b->rows) return 1;
    if (a->cols != b->cols) return 1;
    for (size_t i = 0; i < a->rows * a->cols; i++) {
        a->data[i] = a->data[i] * b->data[i];
    }
    return 0;
}

Matrix *mat_mul(Matrix *a, Matrix *b, Matrix *out) {
    if (a->cols != b->rows) return NULL;
    if (!out) {
        out = mat_create(a->rows, b->cols);
    } else {
        if (out->rows != a->rows) return NULL;
        if (out->cols != b->cols) return NULL;
    }
    for (size_t i = 0; i < out->rows; i++) {
        for (size_t j = 0; j < out->cols; j++) {
            for(size_t k = 0; k < a->cols; k++) {
                float x1 = a->data[i * a->cols + k];
                float x2 = b->data[k * b->cols + j];
                out->data[i * out->cols + j] += x1 * x2;
            }
        }
    }
    return out;
}

Matrix *mat_transpose(Matrix *m) {
    Matrix *m_t = mat_create(m->cols, m->rows);
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            m_t->data[j * m->rows + i] = m->data[i * m->cols + j];
        }
    }
    return m_t;
}

Matrix *mat_from_array(float *a, size_t size, Matrix *out) {
    if (!out) {
        out = mat_create(size, 1);
    } else {
        if (out->rows != size) return NULL;
        if (out->cols != 1) return NULL;
    }
    for (size_t i = 0; i < size; i++) {
        out->data[i] = a[i];
    }
    return out;
}

