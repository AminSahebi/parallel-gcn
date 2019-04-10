#include "module.h"
#include <cstdlib>
#include <cmath>

/* TODO: Implement Matmul, SparseMatmul and GraphSum
 * Matmul = dense-dense matrix multiplication
 * SparseMatmul = sparse-dense matrix multiplication (used in the first layer since the input data is sparse)
 * GraphSum = left-multiplying the normalized Laplacian matrix (try to use more specialized method than sparse matrix multiplication!)
 */ 

CrossEntropyLoss::CrossEntropyLoss(Variable *logits, int *truth, float *loss, int num_classes) {
    this->logits = logits;
    this->truth = truth;
    this->loss = loss;
    this->num_classes = num_classes;
}

void CrossEntropyLoss::forward(bool training) {
    loss = 0;
    for(int i = 0; i < logits->size / num_classes; i++) {
        if (truth[i] < 0) continue;
        float* logit = &logits->data[i * num_classes];
        float sum_exp = 0.0;
        for(int j = 0; j < num_classes; j++)
            sum_exp += logit[j];
        if(training) {
            for(int j = 0; j < num_classes; j++) {
                float prob = expf(logit[j]) / sum_exp;
                logits->grad[i * num_classes + j] = prob;
            }
        }
        loss += logf(sum_exp) - logit[truth[i]];
        if(training) logits->grad[i * num_classes + truth[i]] -= 1.0;
    }
}

void CrossEntropyLoss::backward() {
}

ReLU::ReLU(Variable *in) {
    this->in = in;
    mask = new bool[in->size];
}

ReLU::~ReLU(){
    delete[] mask;
}

void ReLU::forward(bool training) {
    for (int i = 0; i < in->size; i++) {
        bool drop = in->data[i] > 0;
        if (training) mask[i] = drop;
        in->data[i] *= drop;
    }
}

void ReLU::backward() {
    for (int i = 0; i < in->size; i++)
        in->grad[i] *= mask[i];
}

Dropout::Dropout(Variable *in, float p) {
    this->in = in;
    this->p = p;
    mask = new bool[in->size];
}

Dropout::~Dropout(){
    delete[] mask;
}

void Dropout::forward(bool training) {
    const int threshold = int(p * RAND_MAX);
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->size; i++) {
        bool drop = rand() < threshold;
        if (training) mask[i] = drop;
        in->data[i] = drop ? 0 : in->data[i] * scale;
    }
}

void Dropout::backward() {
    float scale = 1 / (1 - p);
    for (int i = 0; i < in->size; i++)
        in->grad[i] *= mask[i] ? 0 : scale;
}