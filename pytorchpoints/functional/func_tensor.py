import itertools
import numpy as np
import torch

def dict_to_cuda(input_dict):
    for key in input_dict:
        if isinstance(input_dict[key], list):
            input_dict[key] = [ val.cuda() for val in input_dict[key]]
        else:
            input_dict[key] = input_dict[key].cuda()
            
def dict_as_tensor(input_dict):
    for key in input_dict:
        if isinstance(input_dict[key], str):
            continue
        elif isinstance(input_dict[key], list):
            input_dict[key] = [torch.as_tensor(x) for x in input_dict[key]]
        else:
            input_dict[key] = torch.as_tensor(input_dict[key])