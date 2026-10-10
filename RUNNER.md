# GPU runner setup (one time, on the training machine)

The training machine runs `v4/worker.py`. It polls this repo every minute, runs each new
job in `jobs/`, and pushes logs, metrics and prediction files to the `runner-results` branch.
Nobody needs to copy files by hand. Raw competition data is never pushed.

1. Accept the GitHub invitation to `harvrennix/dsc-hackathon-v4`. Make sure `git push` works
   from this machine (for example `gh auth login`, or Git Credential Manager on Windows).
2. Clone into a folder used only by the worker. The worker hard-resets tracked files to the
   branch on every poll, so don't edit code in this folder.

   ```sh
   git clone https://github.com/harvrennix/dsc-hackathon-v4 hackathon-runner
   cd hackathon-runner
   ```

3. Put `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle competition in `data/`.
4. Install Python 3.10+ and the dependencies. For PyTorch with CUDA on an RTX 4060, install
   torch from pytorch.org's CUDA wheel index first, then the rest:

   ```sh
   python -m pip install -r requirements.txt
   ```

5. Start the worker and leave it running (keep the machine awake):

   ```sh
   python v4/worker.py
   ```

Check it is alive: the `runner-results` branch gets `worker_info.json` and `heartbeat.json`
within a minute. To stop it, press Ctrl+C. Restarting is safe: finished jobs are not rerun.

## Driving it (from the other machine)

```sh
git remote add runner https://github.com/harvrennix/dsc-hackathon-v4.git   # once
python v4/channel.py submit NAME [--collect GLOB ...] -- script.py [args...]
python v4/channel.py status
python v4/channel.py wait 0002_baseline
python v4/channel.py pull 0002_baseline
```

Security note: the worker runs the Python scripts that collaborators push to this private
repo (no shell, and only `.py` files inside the repo). Only grant write access to people you trust.
