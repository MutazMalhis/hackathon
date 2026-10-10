import sys, time, numpy as np
from pathlib import Path
print("selftest args", sys.argv[1:])  # channel smoke test: python v4/channel.py submit selftest -- v4/_selftest.py 5
np.savez('outputs/v4/selftest.npz', a=np.arange(3)); time.sleep(int(sys.argv[1]) if len(sys.argv)>1 else 1); print('ok')
