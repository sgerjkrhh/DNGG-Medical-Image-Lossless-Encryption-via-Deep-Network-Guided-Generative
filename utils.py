import torch
import numpy as np
import concurrent.futures

from math import cos, sqrt, pi

from torch import Tensor

def process_iterations(a0, p0, start, end):
    ai_part = []
    a0_part = a0.clone()

    for _ in range(start, end):
        a1 = a0_part.clone()
        mask1 = a0_part < p0
        mask2 = (a0_part >= p0) & (a0_part < 0.5)
        mask3 = a0_part >= 0.5

        a1[mask1] = a0_part[mask1] / p0[mask1]
        a1[mask2] = (a0_part[mask2] - p0[mask2]) * (0.5 - p0[mask2])
        a1[mask3] = 1 - a0_part[mask3]

        ai_part.append(a1)
        a0_part = a1.clone()

    return ai_part

def create_pwlcm_paras_metrix(a0, p0, size):
    ai = []
    chunk_size = size // 4  # 将任务分割成 4 个块

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_iterations, a0, p0, i*chunk_size, (i+1)*chunk_size) for i in range(4)]

        for future in concurrent.futures.as_completed(futures):
            ai.extend(future.result())

    ai_tensor = torch.stack(ai)
    sorted_indices = torch.argsort(ai_tensor, dim=0).transpose(1, 0)
    
    return sorted_indices.int()

def create_pwlcm_paras(a0, p0, size):
    ai = []
    for _ in range(size):
        if 0 <= a0 < p0:
            a0 = a0/p0
        elif a0 < 0.5:
            a0 = (a0-p0)*(0.5-p0)
        else:
            a0 = 1-a0
        ai.append(a0)
    return np.argsort(ai)

def create_pwlcm_paras_reverse(a0, p0, size):
    ai = create_pwlcm_paras(a0, p0, size)
    return np.argsort(ai)

def to_binary(x: Tensor) -> Tensor:
    x = x.clone()
    binary_repr = torch.tensor([], dtype=torch.float32).to(x.device)
    for _ in range(8):
        binary_repr = torch.cat((binary_repr, (x % 2).unsqueeze(0)), dim=0)
        x = x // 2
        
    return binary_repr

def to_uint8(x: Tensor):
    binary_repr = []
    for i in range(8):
        binary_repr.append(x[i])
        x = x * 2

    return sum(binary_repr)

def approx_xor(a: Tensor, b: Tensor):
    a, b = a.squeeze(), b.squeeze()
    if len(a.shape) == 4 and len(b.shape) == 3:
        a = a.permute(1, 0, 2, 3)
        return ((a + b) - 2 * (a * b)).permute(1, 0, 2, 3)
    else:
        return (a + b) - 2 * (a * b)
    
    
def create_dct_para(size):
    return [[cos(i * (j + 0.5) * pi / size) / sqrt(size / 2) if i else cos(i * (j + 0.5) * pi / size) / sqrt(size)
            for j in range(size)]
            for i in range(size)]



























if __name__ == '__main__':
    a = np.argsort(create_pwlcm_paras(0.5, 0.3, 100))
    b = np.arange(100)

    print(b[a])
    print(b[a][a])
