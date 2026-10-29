"""Resumable job runs: every finished job is appended to a JSON-lines file at once (flushed and fsynced); a restart
skips the jobs already in the file. A killed run (OOM, crash, stop) therefore loses only the jobs in progress.

Usage:
    from resumable import run_jobs, save_state, load_state
    results = run_jobs(jobs, fn, 'calc/r999_jobs.jsonl', processes=3)
    # jobs: list of JSON-serialisable job descriptions (tuples, dicts, numbers, strings)
    # fn(job) -> JSON-serialisable result; must be a module-level function (multiprocessing)
    # results: dict {key(job): result}, including the jobs finished by earlier runs

For one long computation (an optimisation loop), split it into jobs where possible; otherwise checkpoint its state
every few minutes with save_state(path, dict_of_arrays) and start from load_state(path) when it exists.
"""
import json, os
from multiprocessing import Pool

import numpy as np


def key(job):
    return json.dumps(job, sort_keys=True, default=str)


def done_jobs(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:          # a line cut off by the kill
                    continue
                done[r['key']] = r['result']
    return done


def _call(args):
    fn, job = args
    return key(job), fn(job)


def run_jobs(jobs, fn, path, processes=1):
    done = done_jobs(path)
    todo = [j for j in jobs if key(j) not in done]
    print(f'{path}: {len(done)} jobs done earlier, {len(todo)} to run', flush=True)
    with open(path, 'a') as out:
        if out.tell() > 0:                  # end a line cut off by a kill, so the next record starts clean
            with open(path, 'rb') as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b'\n':
                    out.write('\n')
        def record(k, res):
            out.write(json.dumps({'key': k, 'result': res}, default=float) + '\n')
            out.flush(); os.fsync(out.fileno())
            done[k] = res
            print('done', k, flush=True)
        if processes > 1 and len(todo) > 1:
            with Pool(processes) as pool:
                for k, res in pool.imap_unordered(_call, [(fn, j) for j in todo]):
                    record(k, res)
        else:
            for j in todo:
                record(*_call((fn, j)))
    return done


def save_state(path, state):
    """Atomic checkpoint of a dict of arrays/numbers (written to a temp file, then renamed)."""
    tmp = path + '.tmp.npz'
    np.savez(tmp, **state)
    os.replace(tmp, path)


def load_state(path):
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
