/**
 * @author      : t-hara (t-hara@$HOSTNAME)
 * @file        : mlp
 * @created     : 金曜日 12 05, 2025 00:50:05 JST
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "mlp_params.h"

#define ROUND_CONST (1 << (FXP_VALUE - 1))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))


static void standard_scaler(const int64_t *x, int32_t *y, const int64_t *mean, const int64_t *scale, const int32_t N) {
#pragma clang loop unroll(full)
  for (int32_t i = 0; i < N; ++i) {
    if (mean[i] > x[i])
      y[i] = -(int32_t)(((int64_t)mean[i] - (int64_t)x[i]) * (1 << FXP_VALUE) / (int64_t)scale[i]);
    else
      y[i] = (int32_t)(((int64_t)x[i] - (int64_t)mean[i]) * (1 << FXP_VALUE) / (int64_t)scale[i]);
  }
}

static void linear_layer(const int32_t *w, const int32_t *x, int32_t *y, const int32_t *b, const int32_t M, const int32_t N) {
    int64_t accumulator;
#pragma clang loop unroll(full)
    for (int32_t i = 0; i < M; ++i) {
      accumulator = 0;
      for (int32_t j = 0; j < N; ++j)
        accumulator += ((int64_t)w[i * N + j] * x[j]);
      accumulator = (accumulator + ROUND_CONST) >> FXP_VALUE;
      accumulator += (int64_t)b[i];
      y[i] = (int32_t)accumulator;
    }
}

static void relu(int32_t *tensor, const int32_t size) {
#pragma clang loop unroll(full)
  for (int32_t i = 0; i < size; i++)
    tensor[i] = MAX(tensor[i], 0);
}

void mlp(const int64_t *x, unsigned int *class_indices) {
  int32_t x_scaled[N0] = {0};
  int32_t layer_0_output[N1] = {0};
  int32_t layer_1_output[N2] = {0};
  int32_t layer_2_output[N3] = {0};
  standard_scaler(x, x_scaled, mean, scale, N0);
  linear_layer(layer_0_weight, x_scaled, layer_0_output, layer_0_bias, N1, N0);
  relu(layer_0_output, N1);
  linear_layer(layer_1_weight, layer_0_output, layer_1_output, layer_1_bias, N2, N1);
  relu(layer_0_output, N2);
  linear_layer(layer_2_weight, layer_1_output, layer_2_output, layer_2_bias, N3, N2);
  int ret = (layer_2_output[0] > layer_2_output[1]) ? 0 : 1;
  class_indices[0] = ret;
}
