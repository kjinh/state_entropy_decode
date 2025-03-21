import os
import socket
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from vllm import LLM, SamplingParams
import uvicorn
import gc
import torch
import random

node_ip = socket.gethostbyname(socket.gethostname())
num_gpus = torch.cuda.device_count()
app = FastAPI()

ip_path = "~/state_entropy_decode/src/sal/gen_ip.txt"
ip_path = os.path.expanduser(ip_path)

with open(ip_path, "w") as f:
    f.write(node_ip)
print(f"gen server running on: {node_ip}")

model_name = 'meta-llama/Llama-3.2-1B-Instruct' # 'intfloat/e5-large-v2'
# model_name = 'jinaai/jina-embeddings-v2-base-en'

llm_generate = LLM( # 2~3GB
    model='meta-llama/Llama-3.2-1B-Instruct',
    tensor_parallel_size=num_gpus,
    gpu_memory_utilization=0.35,
)

@app.post("/v1/generateText")
async def generate(request: Request) -> Response :
    request_dict = await request.json()
    prompts = request_dict.pop("prompts") 
    sampling_dict = request_dict.pop("samplings")
    samplings = SamplingParams(**sampling_dict)
    outputs_bef = llm_generate.generate(prompts, samplings, use_tqdm=False)
    outputs = [{'text': output.outputs[0].text,
                'stop_reason': output.outputs[0].stop_reason,
                'log_prob': output.outputs[0].cumulative_logprob} for output in outputs_bef]
    ret = {"result": outputs} # 1D List
    del outputs
    if random.random() < 0.1 : 
        print("Clean!")
        gc.collect()
        for i in range(num_gpus) :
            with torch.cuda.device(f'cuda:{i}') :
                torch.cuda.empty_cache()
                # Reset CUDA device to fully clear memory
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()  # Wait for all streams on the current device
    return JSONResponse(ret)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1110)
# tokenizer = llm.get_tokenizer()
# [good_token_id, bad_token_id, step_tag_id] = tokenizer.encode(f"{good_token} {bad_token} {step_tag}")[1:] # [648, 387]

# @app.post("/v1/generateText")
# async def generateText(request: Request) -> Response:
#     request_dict = await request.json()
#     prompt = request_dict.pop("prompts")
#     outputs = llm.generate(prompt, sampling_params)
#     results = []
#     for output in outputs:
#         result_tmp = []
#         prompt_logprobs = output.prompt_logprobs
#         all_tokens = output.prompt_token_ids
#         tag_token_index = [i for i, token in enumerate(all_tokens) if token == step_tag_id]
#         for token_index in tag_token_index:
#             logprobs = prompt_logprobs[token_index]
#             good_score = 0
#             bad_score = 0
#             if good_token_id in logprobs:
#                 good_score = logprobs[good_token_id].logprob
#             if bad_token_id in logprobs:
#                 bad_score = logprobs[bad_token_id].logprob
                
#             normalized_good_score = torch.softmax(torch.tensor([good_score, bad_score]).float(), dim=0)[0].item()
#             result_tmp.append(normalized_good_score)
#         results.append(result_tmp)
#     ret = {"result": results}
#     del outputs, results
#     torch.cuda.empty_cache()
#     # Reset CUDA device to fully clear memory
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.synchronize()  # Wait for all streams on the current device
    
#     return JSONResponse(ret)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)