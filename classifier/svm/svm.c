/**
 * @file        : svm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "svm_params.h"

void svm(const int64_t *x, unsigned int *class_indices) {
  int argmax = 0;

  if (OUTPUT_DIM == 2 && SVM_NUM_CLASSIFIERS == 1) {
    __int128 score = svm_intercept[0];
    for (int i = 0; i < NUM_FEATURES; i++) {
      score += (svm_coef[i] * x[i]) / SVM_SCALE;
    }
    class_indices[0] = (score >= 0) ? svm_classes[1] : svm_classes[0];
    return;
  }

  __int128 max_score = svm_intercept[0];
  for (int i = 0; i < NUM_FEATURES; i++) {
    max_score += (svm_coef[i] * x[i]) / SVM_SCALE;
  }

  for (int c = 1; c < OUTPUT_DIM; c++) {
    __int128 score = svm_intercept[c];
    for (int i = 0; i < NUM_FEATURES; i++) {
      score += (svm_coef[c * NUM_FEATURES + i] * x[i]) / SVM_SCALE;
    }
    if (score > max_score) {
      max_score = score;
      argmax = c;
    }
  }

  class_indices[0] = svm_classes[argmax];
}
