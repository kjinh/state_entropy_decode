import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm.auto import tqdm

from parser import *
from grader import *
from utils import load_jsonl

import os

# This code snippet is iterating over a list of result paths and processing JSON files within those
# paths. It loads the JSON files, parses the ground truth data, extracts predictions, and then
# evaluates the predictions using a `math_equal_process` function in parallel using a `ProcessPool`.

results_paths = ['Llama3.1-8B-PRM-Deepseek-Data_state_trajectory_scheduler_constant_n_4_svd_refine']
default_path = '~/state_entropy_decode/results_0322/sa_0.005_model_RLHFlow'
default_path = os.path.expanduser(default_path)
print(os.getcwd())
for result_path in results_paths :
    print(result_path)
    lists = os.listdir(os.path.join(default_path, result_path))
    samples = []
    for file_path in lists :
        samples.extend(load_jsonl(os.path.join(default_path, result_path, file_path)))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]
    # parse gt
    for sample in samples:
        # print(sample['pred'])
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, 'math')
        sample['pred'] = extract_answer_map(sample, 'math500k')['pred']
    params = [(idx, sample['level'], sample['pred'], sample['gt']) for idx, sample in enumerate(samples)]
    # samples = samples.map(extract_answer_map, fn_kwargs={"data_name": 'math500k'}, desc="Parsing predictions", num_proc=4, load_from_cache_file=False)
    # params = [(idx, pred, gt) for idx, pred, gt in zip(samples['idx'], samples['pred'], samples['gt'])]

    all_scores = []
    level_scores = {f'Level {i}':[] for i in range(1, 6)}
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    # all_scores.append(result.values(0))
                    for k, v in result.items() :
                        level_scores[k].append(v)
                        all_scores.append(v)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    all_scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 
    mean_score = np.mean(all_scores) * 100
    level_mean_score = {k:np.mean(v)*100 for k, v in level_scores.items()}
    import json
    result_json = {
        "num_samples": len(samples),
        "num_scores": len(all_scores),
        "level_scores": json.dumps(level_mean_score),
        "timeout_samples": timeout_cnt,
        "acc": mean_score
    }

    print(result_json)