import os
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from vllm import LLM, SamplingParams
import uvicorn
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]= "4, 5"

app = FastAPI()

llm = LLM(
    model='peiyi9979/math-shepherd-mistral-7b-prm',
    # dtype='float16',
    tensor_parallel_size=2,
    gpu_memory_utilization=0.4,
    max_model_len=4096
)

sampling_params = SamplingParams(
    logprobs=20,
    prompt_logprobs=20,
)

good_token = '+'
bad_token = '-'
step_tag = 'ки'

tokenizer = llm.get_tokenizer()
[good_token_id, bad_token_id, step_tag_id] = tokenizer.encode(f"{good_token} {bad_token} {step_tag}")[1:] # [648, 387]

@app.post("/v1/generateText")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompts")
    outputs = llm.generate(prompt, sampling_params)
    results = []
    for output in outputs:
        result_tmp = []
        prompt_logprobs = output.prompt_logprobs
        # print(prompt_logprobs)
        all_tokens = output.prompt_token_ids
        tag_token_index = [i for i, token in enumerate(all_tokens) if token == step_tag_id]
        for token_index in tag_token_index:
            logprobs = prompt_logprobs[token_index]
            good_score = 0
            bad_score = 0
            # print(logprobs, token_index)
            if good_token_id in logprobs:
                good_score = logprobs[good_token_id].logprob
            if bad_token_id in logprobs:
                bad_score = logprobs[bad_token_id].logprob
                
            normalized_good_score = torch.softmax(torch.tensor([good_score, bad_score]).float(), dim=0)[0].item()
            result_tmp.append(normalized_good_score)
        results.append(result_tmp)
    ret = {"result": results}
    del outputs
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    # Reset CUDA device to fully clear memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # Wait for all streams on the current device
    
    return JSONResponse(ret)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)