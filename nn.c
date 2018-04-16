#include "nn.h"

NeuralNetwork *create_nn(uint ni, uint nh, uint no) {
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    nn->lr = 0.1;
    nn->ni = ni;
    nn->nh = nh;
    nn->no = no;
    nn->i = create_matrix(ni, 1);
    nn->h = create_matrix(nh, 1);
    nn->o = create_matrix(no, 1);
    nn->w_ih = create_matrix(nh, ni);
    nn->w_ho = create_matrix(no, nh);
    nn->b_h = create_matrix(nh, 1);
    nn->b_o = create_matrix(no, 1);
    randomize(nn->w_ih);
    randomize(nn->w_ho);
    randomize(nn->b_h);
    randomize(nn->b_o);
    return nn;
}

int destroy_nn(NeuralNetwork *nn) {
    if (!nn) return 1;
    destroy_matrix(nn->i);
    destroy_matrix(nn->h);
    destroy_matrix(nn->o);
    destroy_matrix(nn->w_ih);
    destroy_matrix(nn->w_ho);
    destroy_matrix(nn->b_h);
    destroy_matrix(nn->b_o);
    free(nn);
    return 0;
}

Matrix *sigmoid(Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        m->data[i] = (float)(1 / (1 + exp(-x)));
    }
    return m;
}

Matrix *gradients(Matrix *m) {
    // sigmoid(Wx+b) * (1 - sigmoid(Wx+b))
    Matrix *grad = create_matrix(m->rows, m->cols);
    for(size_t i = 0; i < m->rows * m->cols; i++)
        grad->data[i] = 1 - m->data[i];
    m = mult_elem(m, grad);
    destroy_matrix(grad);
    return m;
}

void nn_train(NeuralNetwork *nn, float *ay) {
    // adjust w_ho
    Matrix *y = matrix_from_array(ay, nn->no, NULL);
    Matrix *err_o = sub_elem(y, nn->o);
    // calculate the gradients
    Matrix *grad_o = gradients(nn->o);
    grad_o = mult_elem(grad_o, err_o);
    grad_o = mult_scalar(grad_o, nn->lr);
    // calculate deltas
    Matrix *h_t = transpose(nn->h, NULL);
    Matrix *dw_ho = mult_mat(grad_o, h_t, NULL);
    // update the weigths and biases
    nn->w_ho = add_elem(nn->w_ho, dw_ho);
    nn->b_o = add_elem(nn->b_o, grad_o);

    // adjust w_ih
    // skip normalizing the weights for this simple network
    // e.g. just W[1][1] instead of W[1][1] / Sum(j, W[1][j])
    Matrix *w_ho_t = transpose(nn->w_ho, NULL);
    Matrix *err_h = mult_mat(w_ho_t, err_o, NULL);
    // calculate the gradients
    Matrix *grad_h = gradients(nn->h);
    grad_h = mult_elem(grad_h, err_h);
    grad_h = mult_scalar(grad_h, nn->lr);
    // calculate the deltas
    Matrix *i_t = transpose(nn->i, NULL);
    Matrix *dw_ih = mult_mat(grad_h, i_t, NULL);
    // adjust the weights and biases
    nn->w_ih = add_elem(nn->w_ih, dw_ih);
    nn->b_h = add_elem(nn->b_h, grad_h);

    // cleanup
    destroy_matrix(err_o);
    destroy_matrix(h_t);
    destroy_matrix(dw_ho);
    destroy_matrix(w_ho_t);
    destroy_matrix(err_h);
    destroy_matrix(i_t);
    destroy_matrix(dw_ih);
}

Matrix *nn_predict(NeuralNetwork *nn, float *ax) {
    // sigmoid(W_ih * h + b_h)
    nn->i = matrix_from_array(ax, nn->ni, nn->i);
    nn->h = mult_mat(nn->w_ih, nn->i, nn->h);
    nn->h = add_elem(nn->h, nn->b_h);
    nn->h = sigmoid(nn->h);
    // sigmoid(W_ho * o + b_o)
    nn->o = mult_mat(nn->w_ho, nn->h, nn->o);
    nn->o = add_elem(nn->o, nn->b_o);
    nn->o = sigmoid(nn->o);
    return nn->o;
}

