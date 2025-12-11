import pandas as pd
import numpy as np
import torch
DECIMAL_PART = 16
SCALING_FACTOR = 2 ** DECIMAL_PART

def fwrite_params(f, name, params):
    print(name)
    f.write(f"static const int32_t {name}[{params.shape[0]}] = {{")
    for idx in range(params.shape[0]):
        if idx == 0:
            f.write(f'{params[idx]}')
        else:
            f.write(f', {params[idx]}')
    f.write("};\n")

if __name__ == "__main__":
    saved_stats = torch.load("mlp_fixed.pth")
    state_dict = saved_stats['model_state_dict']
    print(saved_stats)
    mean = saved_stats['mean']
    scale = saved_stats['scale']

    weights = {k: v for k, v in state_dict.items() if 'weight' in k}
    bias = {k: v for k, v in state_dict.items() if 'bias' in k}
    with open("mlp_params.h", "w") as f:
        f.write('#ifndef MLP_PARAMS_BPF_H\n')
        f.write('#define MLP_PARAMS_BPF_H\n\n')
        f.write(f'#define FXP_VALUE {DECIMAL_PART}\n\n')
        for i, (name, params) in enumerate(weights.items()):
            if i == 0:
                f.write(f"#define N{i} {params.shape[1]}\n")
            f.write(f"#define N{i+1} {params.shape[0]}\n")
        for i, (name, params) in enumerate(weights.items()):
            params = params.flatten()
            params = (params * SCALING_FACTOR).round().cpu().numpy().astype(np.int64)
            fwrite_params(f, f"layer_{i}_weight", params)
        for i, (name, params) in enumerate(bias.items()):
            params = params.flatten()
            params = (params * SCALING_FACTOR).round().cpu().numpy().astype(np.int64)
            fwrite_params(f, f"layer_{i}_bias", params)
        mean = (saved_stats['mean'] * SCALING_FACTOR).round().astype(np.int64)
        f.write(f"static const int64_t mean[{len(mean)}] = {{")
        for idx in range(len(mean)):
            f.write(f"{mean[idx]}")
            if idx < len(mean) - 1:
                f.write(", ")
        f.write("};\n")
        scale = (saved_stats['scale'] * SCALING_FACTOR).round().astype(np.int64)
        f.write(f"static const int64_t scale[{len(scale)}] = {{")
        for idx in range(len(scale)):
            f.write(f"{scale[idx]}")
            if idx < len(scale) - 1:
                f.write(", ")
        f.write("};\n")
        f.write('\n#endif // MLP_PARAMS_BPF_H\n')
