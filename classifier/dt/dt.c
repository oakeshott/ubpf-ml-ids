/**
 * @author      : t-hara (t-hara@$HOSTNAME)
 * @file        : dt
 * @created     : 木曜日 12 04, 2025 19:46:23 JST
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "dt_params.h"
#define MAX_TREE_DEPTH 10
#define TREE_LEAF -1
#define TREE_UNDEFINED -2
#define NUM_FEATURES 12
#define abs(x) ((x)<0 ? -(x) : (x))

void dt(const int64_t *x, unsigned int *class_indices) {
  int current_node = 0;
  /* for (int i = 0; i < NUM_FEATURES; i++) */
  /*   printf("%lld, ", x[i]); */
  /* printf("\n"); */
  for (int i = 0; i < MAX_TREE_DEPTH; i++) {
    int64_t current_left_child  = children_left[current_node];
    int64_t current_right_child = children_right[current_node];
    int64_t current_feature     = features[current_node];
    int64_t current_threshold   = threshold[current_node];
    if (current_left_child == TREE_LEAF || current_feature == TREE_UNDEFINED) {
      break;
    } else {
      int64_t current_feature_value = x[current_feature];
      if (current_feature_value <= current_threshold) {
        current_node = (int)current_left_child;
      } else {
        current_node = (int)current_right_child;
      }
    }
  }
  class_indices[0] = value[current_node];
}
