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

from itertools import accumulate

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sal.config import Config
import requests
import json

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    url: str,
    headers,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        prompts = {'prompts': inputs_batch}
        step_scores = json.loads(requests.post(url, headers=headers, data=json.dumps(prompts)).text)['result']

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch
        torch.cuda.empty_cache()

    return output_scores


class PRM:
    def __init__(self, search_config: Config, ip, **model_kwargs):
        self.search_config = search_config
        self.headers = {"Content-Type": "application/json"}
        self.url = f"http://{ip}:8000/v1/generateText"
        # self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


class MathShepherd(PRM):
    # def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    #     model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
    #     _supports_flash_attn_2 = True
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     # For batched inference
    #     tokenizer.pad_token = tokenizer.eos_token
        
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         device_map="auto",
    #         attn_implementation="sdpa",
    #         torch_dtype=torch.float16,
    #     ).eval()
    #     return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.url,
            self.headers,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # print(cumulative_lengths, output_scores)
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]
        # print(output_scores, outputs)

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores


class RLHFFlow(PRM):
    # def load_model_and_tokenizer(
    #     self, **model_kwargs
    # ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    #     )
    #     model = AutoModelForCausalLM.from_pretrained(
    #         "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
    #         device_map="auto",
    #         torch_dtype=torch.bfloat16,
    #         **model_kwargs,
    #     ).eval()
    #     tokenizer.padding_side = "right"
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    #     plus_tag_id = tokenizer.encode("+")[-1]
    #     minus_tag_id = tokenizer.encode("-")[-1]
    #     self.candidate_tokens = [plus_tag_id, minus_tag_id]

    #     return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = False,
        # batch_size=8,
    ) -> list[list[float]]:
        if self.search_config.prm_batch_size > 1:
            return self._score_batched(questions, outputs, batch_size=self.search_config.prm_batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.rstrip('\n\n').split('\n\n')
                # ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                single_step_score.append(json.loads(requests.post(self.url, headers=self.headers, data=json.dumps({'chat' : [conversation]})).text)['result'][0])
                    # input_ids = self.tokenizer.apply_chat_template(
                    #     conversation, return_tensors="pt"
                    # ).to(self.model.device)
                    # with torch.no_grad():
                    #     logits = self.model(input_ids).logits[
                    #         :, -3, self.candidate_tokens
                    #     ]  # simple version, the +/- is predicted by the '-3' position
                    #     step_scores = logits.softmax(dim=-1)[
                    #         :, 0
                    #     ]  # 0 means the prob of + (1 mean -)
                    #     # print(scores)
                    #     single_step_score.append(
                    #         step_scores[0]
                    #         .detach()
                    #         .to("cpu", dtype=torch.float32)
                    #         .item()
                    #     )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        headers = {"Content-Type": "application/json"}
        url = self.url
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        output_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                single_score = []
                # split after rstrip to remove the final newline
                ans_list = ans.rstrip('\n\n').split('\n\n')
                # ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
            conversations.append(conversation)
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            # inputs_batch = self.tokenizer.apply_chat_template(
            #     convs_batch, padding=True, return_tensors="pt"
            # ).to(self.model.device)
            # inputs2_batch = self.tokenizer.apply_chat_template(
            #     convs2_batch, padding=True, return_tensors="pt"
            # ).to(self.model.device)
            # assert inputs_batch.shape == inputs2_batch.shape
            # with torch.no_grad():
            #     logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
            #     scores = logits.softmax(dim=-1)[
            #         :, :, 0
            #     ]  # 0 means the prob of + (1 mean -)

            #     for i in range(len(convs_batch)):
            #         # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
            #         step_scores_flat = scores[i, :-1][
            #             inputs2_batch[i, 1:] == special_tok_id
            #         ].tolist()
            #         output_scores.append(step_scores_flat)
            # Request to vLLM server
            output_score = json.loads(requests.post(url, headers=headers, data=json.dumps({'chat' : convs_batch})).text)['result']
            output_scores.extend(torch.tensor(output_score).reshape(-1, 1).tolist())
        # reshape the output scores to match the input
        reshaped_output_scores = []
        # print(output_scores, question, outputs)
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for _ in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)
        del output_scores, conversations
        # torch.cuda.empty_cache()
        return reshaped_output_scores


def load_prm(config: Config, ip: str) -> PRM:
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config, ip)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")
