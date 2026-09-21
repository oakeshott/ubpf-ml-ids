/**
 * @file        : xgb
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "xgb_params.h"

#define TREE_LEAF -1
#define TREE_UNDEFINED -2

void xgb(const int64_t *x, unsigned int *class_indices) {
  int64_t accumulator = XGB_BASE_SCORE;

  for (int j = 0; j < N_ESTIMATORS; j++) {
    int current_node = 0;
    for (;;) {
      int64_t current_left_child  = children_left[j * N + current_node];
      int64_t current_right_child = children_right[j * N + current_node];
      int64_t current_feature     = features[j * N + current_node];
      int64_t current_threshold   = threshold[j * N + current_node];

      if (current_right_child == TREE_LEAF || current_left_child == TREE_LEAF ||
          current_threshold == TREE_UNDEFINED || current_feature == TREE_UNDEFINED) {
        break;
      }

      if (current_feature >= 0 && current_feature < NUM_FEATURES) {
        int64_t current_feature_value = x[current_feature];
        if (current_feature_value < current_threshold) {
          current_node = (int)current_left_child;
        } else {
          current_node = (int)current_right_child;
        }
      } else {
        break;
      }
    }
    accumulator += value[j * N + current_node];
  }

  class_indices[0] = (accumulator >= 0) ? xgb_classes[1] : xgb_classes[0];
}
