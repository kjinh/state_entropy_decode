#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, PBE, RMS, build_conv, generate_k_steps

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

def _beam_search(batch_of_prompts, config: Config, prm: PRM, ip: dict[str, str]) -> list[Beam]:
    rms = RMS("cuda:0")
    stat_entropy = PBE(rms, config.embedding_model, "cuda:0", ip["embed"])
    history = []
    sampling_params_dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "stop": ["\n\n"],
        "include_stop_str_in_output": True,
        "n": 1,
    }
    # sampling_params = SamplingParams(
    #     temperature=config.temperature,
    #     max_tokens=config.max_tokens,
    #     top_p=config.top_p,
    #     stop=["\n\n"],
    #     include_stop_str_in_output=True,
    #     n=1,
    # )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                    regenerate=-1
                )
            )
    completed_beams: list[Beam] = []

    for nw in tqdm(range(config.num_iterations), desc=f"Beam search iterations with sa {config.entropy_hyper}"):
        if nw == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if nw == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params_dict = {
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "n": 1,
            }
            # sampling_params = SamplingParams(
            #     temperature=config.temperature,
            #     max_tokens=config.max_tokens,
            #     top_p=config.top_p,
            #     n=1,
            # )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt, config.regenerate_prompt, b.regenerate, b.history)
            for b in active_beams
        ]
        continue_final_message = nw > 0
        add_generation_prompt = nw == 0

        # tokenizer = llm.get_tokenizer()
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if nw == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, sampling_params_dict, 1, ip["gen"]
        )

        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])
            # print(len(beam.next_texts[0]))

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])
                
        # print(len(prompts[0]))
        scores = prm.score(prompts, completions)
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]
        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]
            
        # Replay Buffer add
        flat_agg = np.array(agg_scores).flatten()
        if config.entropy_hyper > 0 :
            if config.state_traj == "trajectory" :
                beams_final_text = [active_beam.current_text for active_beam in active_beams]
            elif config.state_traj == "traj_w_question" :
                # ref. score() in reward_models.py
                beams_final_text = [active_beam.prompt + " " + active_beam.current_text for active_beam in active_beams]
            else :
                beams_final_text = [active_beam.next_texts[0] for active_beam in active_beams]
            # The answer generated in this step should not be compared with the answer before it => pass the parent 
            if nw > 0 and len(active_beams[0].history) > 1:
                parents_collect = {"None": []}
                for i, active_beam in enumerate(active_beams):
                    if len(active_beam.history) <= 1:
                        parents = "None"
                    else :
                        parents = ''.join(active_beam.history[:-1]) if config.state_traj == 'trajectory' else active_beam.history[-2]
                    if parents not in parents_collect.keys() :
                        parents_collect[parents] = []
                    parents_collect[parents].append(i)
                reward_add, sa_records = stat_entropy(beams_final_text, parents_collect)
                reward_add = reward_add.flatten().to("cpu")
                implict_reward = max(0.0, config.entropy_hyper * (11 - nw)/10) if config.scheduler == "linear" else config.entropy_hyper
                flat_agg = (torch.tensor(flat_agg) + reward_add * implict_reward).tolist()
                # print(f"PRM Reward: {flat_agg}\nSA Reward: {reward_add.tolist()}")
                # for i, reward in enumerate(reward_add) :
                #     if torch.is_tensor(reward) :
                #         flat_agg[i] += implict_reward * (reward.item())
                #         # if config.scheduler == "linear" :
                #         #     print(implict_reward)
                #     else :
                #         flat_agg[i] += implict_reward * reward
                # print(reward_add)
                history.extend(sa_records)
            
            # Replay buffer update (the process of putting the data generated in this step into the replay buffer at the very end)
            stat_entropy.update(beams_final_text)
        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(flat_agg)[
            -(config.n // config.beam_width) :
        ]
        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True
        del templated_convs, gen_results

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams
    del active_beams, rms, stat_entropy
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Reset CUDA device to fully clear memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # Wait for all streams on the current device
    return completed_beams, history


def beam_search(examples, config: Config, prm: PRM, ip: dict[str, str]):
    problems = examples["problem"]
    beam_results, history_pers = _beam_search(problems, config, prm, ip)
    # print(history_pers)
    import os
    history_path = '~/state_entropy_decode/history/history_0323_step_no_reduction_05'
    history_path = os.path.expanduser(history_path)
    
    if not os.path.exists(history_path) :
        os.makedirs(history_path)
    file_counts = len(list(os.listdir(history_path)))
    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])
    
    history_pers.append(results)
    import json
    with open(os.path.join(history_path, f'problem_numbers_{file_counts}.json'), "w") as fp:
        json.dump(history_pers, fp)
    return results