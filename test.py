import pprint
import sys
import platform

import torch
import lightgbm
import threadpoolctl

print('version: ', sys.version, flush=True)
print('platform: ', platform.platform(), flush=True)
pprint.pprint(threadpoolctl.threadpool_info())

print('before torch tensor', flush=True)
t = torch.ones(200_000)
print('after torch tensor', flush=True)
