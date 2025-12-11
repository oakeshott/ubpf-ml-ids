/**
 * @author      : t-hara (t-hara@$HOSTNAME)
 * @file        : rf
 * @created     : 木曜日 12 04, 2025 23:32:47 JST
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "rf_params.h"
#define OUTPUT_DIM 2
#define TREE_LEAF -1
#define TREE_UNDEFINED -2
void rf(const int64_t *x, unsigned int *class_indices) {
  /*  */
  /* for (int i = 0; i < NUM_FEATURES; i++) */
  /*   printf("%lld, ", x[i]); */
  /* printf("\n"); */
  /*  */
  int current_node, accumulator[OUTPUT_DIM];
  for (int j = 0; j < OUTPUT_DIM; j++) {
    accumulator[j] = 0;
  }
  for (int j = 0; j < N_ESTIMATORS; j++) {
    current_node = 0;
    for (int i = 0; i < MAX_TREE_DEPTH; i++) {
      int64_t current_left_child  = children_left[j*N + current_node];
      int64_t current_right_child = children_right[j*N + current_node];
      int64_t current_feature     = features[j*N + current_node];
      int64_t current_threshold   = threshold[j*N + current_node];
      if (current_right_child == TREE_LEAF || current_left_child == TREE_LEAF || current_threshold == TREE_UNDEFINED || current_feature == TREE_UNDEFINED) {
        break;
      } else {
        if (current_feature >= 0 && current_feature < NUM_FEATURES ) {
          int64_t current_feature_value = x[current_feature];
          if (current_feature_value <= current_threshold) {
            current_node = (int) current_left_child;
          } else {
            current_node = (int) current_right_child;
          }
        }
      }
    }
    int64_t current_value = value[j*N + current_node];
    accumulator[current_value]++;
  }
  int argmax = 0;
  int max_val = 0;
  for (int j = 0; j < OUTPUT_DIM; j++) {
    if (accumulator[j] > max_val) {
      argmax = j;
      max_val = accumulator[j];
    }
  }
  /* printf("argmax %d\n", argmax); */
  /* printf("%lld\t%d\n", current_value, current_node); */
  class_indices[0] = argmax;
  /* return current_value; */
}
