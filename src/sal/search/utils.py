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
import numpy as np

from dataclasses import dataclass

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import json
import requests
import umap
import warnings
warnings.filterwarnings("ignore")

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"

logger = logging.getLogger()
#API_KEY = os.environ["OPENAI_API_KEY"]
#client = OpenAI()


def build_conv(
    prompt: str, response: str | None, system_prompt: str, regenerate_prompt: str, regenerate_check: int = -1, step_text: list[str] = None
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt}
    ]
    # if wrong_answer != None :
    #     conversation = [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": prompt},
    #     ]
    
    # assistant = "" + str(response + "") + str(wrong_answer + "")
    
    # if assistant != "":
    #     conversation.append({"role": "assistant", "content": response})
    if regenerate_check > 0 :
        start_response = '\n\n'.join(step_text[:regenerate_check])
        conversation.extend([
            {"role": "user", "content": regenerate_prompt.format(regenerate_check=regenerate_check)},
            {"role": "assistant", "content": response},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": start_response}
        ])
    else :
        conversation.append(
            {"role": "user", "content": prompt}
        )

        if response != "":
            conversation.append({"role": "assistant", "content": response})
    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)


@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]]  # all PRM scores
    previous_text: str | None
    pruned: False
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0
    regenerate: int = -1


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None

class refine_result(BaseModel):
    wrong_check: bool
    refine_step: int
    
def refine_answer(
    refine_prompt,
    prompt,
    check_answer: str,
    steps_refine: int
) -> int:
    # completion = client.beta.chat.completions.parse(
    #     model="gpt-4o-2024-11-20",
    #     messages=[
    #         {"role": "system", "content": refine_prompt.format(prompt=prompt, steps_refine= steps_refine)},
    #         {"role": "user", "content": check_answer}
    #     ],
    #     response_format= refine_result
    # )
    # step_checks = completion.choices[0].message.parsed
    # # print(step_checks, type(step_checks))
    # return -1 if step_checks.wrong_check is False else step_checks.refine_step
    pass
    

def generate_k_steps(
    templated_convs,
    lookahead_steps: int,
    sampling_params_dict: dict,
    beam_width: int,
    gen_ip: str
) -> list[Beam]:
    gen_results = []
    headers = {"Content-Type": "application/json"}
    url = f"http://{gen_ip}:1110/v1/generateText"
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params_dict = copy.deepcopy(sampling_params_dict)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params_dict['temperature'] = 0.0
            # gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results))
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]
        request_header = {"prompts": gen_prompts, "samplings": gen_sampling_params_dict}
        llm_outputs = json.loads(requests.post(url, headers=headers, data=json.dumps(request_header)).text)['result']
        # llm_outputs = llm.generate(gen_prompts, gen_sampling_params, use_tqdm=False)
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output["text"]
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output["stop_reason"]
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output["stop_reason"]
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

    outputs: list[Beam] = []

    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            lookahead_texts.append(gen_result.lookahead_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            counter += 1

        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
        )
        outputs.append(beam_result)

    return outputs

'''
Refer to https://deep-ch.medium.com/dimension-reduction-by-whitening-bert-roberta-5e103093f782
Dimension reduction (Bert Whitening)
'''
def transform_and_normalize(vecs, kernel, bias):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).matmul(kernel)
    return normalize(vecs)
    # return normalize(vecs)
    
def normalize(vecs):
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    
def compute_kernel_bias(vecs):
    # vecs = torch.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = torch.cov(vecs.T)
    u, s, _ = torch.linalg.svd(cov)
    W = torch.matmul(u, torch.diag(s**0.5))
    W = torch.linalg.inv(W.T)
    return W, -mu
    
def reduction(input, device) :
    kernel, bias = compute_kernel_bias(input)
    kernel = kernel[:, :512]
    # embeddings = []
    # embeddings = torch.vstack([input])
    embeddings = input
    embeddings = transform_and_normalize(
            embeddings, 
            kernel=kernel,
            bias=bias,
        )
    return embeddings
    # return torch.from_numpy(embeddings)

"""
This function takes a list of input texts, tokenizes them, generates embeddings using a language
model, and returns the sum of embeddings along with the corresponding mask.

:param tokenizer: The `tokenizer` parameter is an instance of a language model tokenizer (LLM) used
for tokenizing input text data
:type tokenizer: LLM
:param model: The `model` parameter in the `embedding_models` function refers to a pre-trained
language model (LLM) that will be used to generate embeddings for the input text. This model is
typically a neural network-based model that has been trained on a large corpus of text data for
tasks such as language
:type model: LLM
:param input: The function `embedding_models` takes the following parameters:
:type input: list[str]
:param device: The `device` parameter in the function `embedding_models` is used to specify whether
the computation should be performed on a CPU or a CUDA-enabled GPU. It is typically passed as an
argument to PyTorch tensor operations to indicate where the tensor should be stored and on which
device the computation should be
:return: The function `embedding_models` returns two tensors: `sum_embeddings` and `sum_mask`.
"""
def embedding_models(
    input: list[str],
    device,
    embed_ip: str,
    model=None,
    tokenizer=None,
    ) :
    total_embeddings = []
    batch_size = 1
    if model is not None :
        with torch.no_grad() :
            for i in range(0, len(input), batch_size) :
                input_batch = input[i : i+batch_size]
                input_token = tokenizer(input_batch, return_tensors="pt", add_special_tokens=True, padding=True).to(device)
                output_state = model(**input_token)
                attention_mask = input_token['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_state[0].size()).float() 
                sum_embeddings = torch.sum(output_state[0] * input_mask_expanded, 1).to(device)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9).to(device)
                total_embeddings.append(sum_embeddings / sum_mask)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                # Reset CUDA device to fully clear memory
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()  # Wait for all streams on the current device
    else :
        headers = {"Content-Type": "application/json"}
        url = f"http://{embed_ip}:1200/v1/embeds"
        with torch.no_grad() :
            # print(max([len(inputs) for inputs in input]))
            for i in range(0, len(input), batch_size) :
                input_batch = input[i : i+batch_size]
                request_header = {"prompts": input_batch}
                llm_embedings = torch.tensor(json.loads(requests.post(url, headers=headers, data=json.dumps(request_header)).text)['result'], device=device)
                total_embeddings.append(llm_embedings)
            # input_token = tokenizer(input_batch, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True).to(device)
            # output_state = model(**input_token)
            # attention_mask = input_token['attention_mask']
            # input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_state[0].size()).float() 
            # sum_embeddings = torch.sum(output_state[0] * input_mask_expanded, 1).to(device)
            # sum_mask = input_mask_expanded.sum(1)
            # sum_mask = torch.clamp(sum_mask, min=1e-9).to(device)
            # total_embeddings.append(sum_embeddings / sum_mask)
        # torch.cuda.empty_cache()
        # output = reduction(sum_embeddings / sum_mask)
    return torch.cat(total_embeddings, dim=0)

def cal_svd(
    source_embeds,
    target_embeds,
    device, 
    test=False
    ) :
    embeddings = torch.cat((source_embeds, target_embeds), dim=0).to(device)
    if test :
        return embeddings
    else :
        input = embeddings.detach().cpu().numpy()
        num_samples = input.shape[0]
        if num_samples < 4 :
            return embeddings
        n_neighbors = min(15, max(2, num_samples - 2))
        n_components = min(128, max(2, num_samples - 2))
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
        return torch.from_numpy(reducer.fit_transform(input)).float().to(device)

class RMS(object):
    """running mean and std """
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon
        self.device = device

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0).to(self.device) - self.M
        new_M = (self.M + delta * bs / (self.n + bs)).to(self.device)
        new_S = ((self.S * self.n + torch.var(x, dim=0).to(self.device) * bs +
                 torch.square(delta) * self.n * bs /
                 (self.n + bs)) / (self.n + bs)).to(self.device)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S
    
# device: 'cuda:0'
class PBE(object):
    """particle-based entropy based on knn normalized by running mean """
    def __init__(self, rms, embed_models, device, ip):
        self.rms = rms
        self.replay_buffer = []
        self.device = device
        self.model = embed_models
        self.emb_model = None
        self.tokenizer = None
        self.ip = ip
        if 'deberta' in self.model :
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.emb_model = AutoModel.from_pretrained(self.model).eval().to(device)
            self.emb_model.eval()
        if 'large' in embed_models or 'v3' in embed_models:
            embed_size = 1024
        elif 'base' in embed_models :
            embed_size = 768
        elif 'deepspeek' in embed_models :
            embed_size = 4096
        else :
            embed_size = 2048
        
        # embed_size = 1024 if 'large' in embed_models else 768
        self.embeddings = torch.empty((0, embed_size), dtype=torch.float16).to(device)
    
    def calculations(self, source_embeddings, target_embeddings, checking) :
        b1, b2 = source_embeddings.size(0), target_embeddings.size(0)
        output = cal_svd(source_embeddings, target_embeddings, self.device, checking).to(self.device)
        source, target = output[:b1, :], output[b1:, :]
        assert (source.size(0) + target.size(0)) == b1+b2, "Not size matchings"
        if b2 == 0:
            return reward
        ### Calculate State Entropy
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
                                target[None, :, :].view(1, b2, -1),
                                dim=-1,
                                p=2).to(self.device)
        # sim_matrix /= self.rms(sim_matrix.reshape(-1, 1))[0]
        # print(target, source)
        # sim_matrix = F.cosine_similarity(target.unsqueeze(0), source.unsqueeze(1), dim=2).to(self.device) # (b1, b2)
        # print(target, source, sim_matrix)
        # print(b1, b2, sim_matrix.shape)
        # print(sim_matrix)
        knn_k = 1
        reward, _ = sim_matrix.topk(min(knn_k, b2), dim=1, largest=False, sorted=True)
        #reward, _ = sim_matrix.topk(1, dim=1, largest=True, sorted=True) # Only Cosine
        if b2 >= knn_k :
            reward /= self.rms(reward)[0]
        # print(reward)
        # reward = reward.reshape(-1, 1)
        reward = reward.reshape((b1, min(knn_k, b2)))
        reward = reward.mean(dim=1, keepdim=True)
        reward = torch.log(reward + 1.0) # (b1, )
        return reward.flatten().to(self.device), sim_matrix.tolist()
    
    def __call__(self, nodes, parent_collect):
        # print(len(self.replay_buffer))
        sa_collections = []
        reward = torch.tensor([0.0] * len(nodes)).to(self.device)
        ### Calculate Embedding & Dimension Reduction
        if len(self.replay_buffer) > 0 : 
            source_embeddings = embedding_models(nodes, self.device, self.ip, self.emb_model, self.tokenizer) # 768 size
            target_embeddings_bef = self.embeddings # 768 size
            for parent, sources in parent_collect.items() :
                if len(sources) == 0 :
                    continue
                target_buffer_bef = copy.deepcopy(self.replay_buffer)
                source_parent = source_embeddings[sources]
                target_embeddings = target_embeddings_bef
                # Find parent in reply buffer
                if parent != 'None' :
                    for idx, strs in enumerate(self.replay_buffer) :
                        if strs == parent :
                            indices = torch.arange(len(self.replay_buffer)).to(self.device)
                            filter_indices = indices[indices != idx]
                            target_embeddings = torch.index_select(target_embeddings_bef, 0, filter_indices).to(self.device)
                            del target_buffer_bef[idx]
                            break
                target_buffer = target_buffer_bef
                # print(target_embeddings_bef)
                reward_parent, sas = self.calculations(source_parent, target_embeddings, True)
                sa_collections.extend([{'source': nodes[source], 'target': target_buffer, 'similarity': sims} for source, sims in zip(sources, sas)])
                reward[sources] = reward_parent
                # for idx, strs in enumerate(self.replay_buffer) :
                #     if strs == parent :
                #         if idx + 1 != len(self.replay_buffer) :
                #             target_embeddings = torch.cat((target_embeddings_bef[:idx, :], target_embeddings_bef[idx+1: , :]), axis = 0)
                #             # target = torch.cat((target[:idx, :], target[idx+1: , :]), axis = 0)
                #         else :
                #             target_embeddings = target_embeddings_bef[:idx, :]
            # print(f"""====
            #       Dimension Reduction Before: {self.calculations(source_embeddings, source_mask, target_embeddings, target_mask, True).to(self.device).tolist()}
            #       Dimension Reduction After: {self.calculations(source_embeddings, source_mask, target_embeddings, target_mask, False).to(self.device).tolist()}
            #         ====""")
            # return self.calculations(source_embeddings, target_embeddings, True).to(self.device)
            # b1, b2 = source_embeddings.size(0), target_embeddings.size(0)
            # source, target = output[:b1, :], output[b1:, :]
            # assert (source.size(0) + target.size(0)) == b1+b2, "Not size matchings"
            # if b2 == 0:
            #     return reward
            # ### Calculate State Entropy
            # # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
            # sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) -
            #                         target[None, :, :].view(1, b2, -1),
            #                         dim=-1,
            #                         p=2).to(self.device)
            # # sim_matrix /= self.rms(sim_matrix.reshape(-1, 1))[0]
            # # print(sim_matrix)
            # reward, _ = sim_matrix.topk(1, dim=1, largest=False, sorted=True)
            # reward /= self.rms(reward)[0]
            # # print(reward)
            # # reward = reward.reshape(-1, 1)
            # reward = reward.reshape((b1, -1))
            # reward = reward.mean(dim=1, keepdim=True)
            # reward = torch.log(reward + 1.0) # (b1, )
        del source_embeddings, target_embeddings_bef, target_embeddings
        return reward, sa_collections
    def __del__(self) :
        # print(self.replay_buffer)
        del self.replay_buffer, self.tokenizer, self.emb_model
        # gc.collect()
        # torch.cuda.empty_cache()
        
    # update Replay Buffer
    def update(self, nodes) :
        # To perform whitening, we need to collect embedding result & attention masks
        sum_embeds = embedding_models(nodes, self.device, self.ip, self.emb_model, self.tokenizer)
        self.embeddings = torch.cat((self.embeddings, sum_embeds), dim=0).to(self.device)
        self.replay_buffer.extend(nodes)
        # self.replay_buffer = torch.cat((self.replay_buffer, ), dim=0)