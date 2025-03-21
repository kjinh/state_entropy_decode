from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from vllm import LLM
import uvicorn
import gc
import torch
import random

num_gpus = torch.cuda.device_count()
app = FastAPI()

model_name = 'meta-llama/Llama-3.2-1B-Instruct' # 'intfloat/e5-large-v2'

llm_embeds = LLM(
    model=model_name,
    tensor_parallel_size=num_gpus,
    gpu_memory_utilization=0.35,
    task="embed"
)
    
@app.post("/v1/embeds")
async def embeds(request: Request) -> Response:
    """
    The function `generateText` processes input data, generates text outputs, and calculates scores
    based on the generated text.
    
    :param request: The `request` parameter is of type `Request`, which is used to represent an incoming
    HTTP request. In this case, it seems like you are expecting a JSON payload in the request body,
    which you are then extracting and processing in your `generateText` function. The function processes
    the JSON payload
    :type request: Request
    :return: The function `generateText` is returning a JSON response containing the "result" key, which
    holds a 2D list of normalized scores calculated based on the input chat data. The normalized scores
    are computed by processing the outputs generated by the `llm.chat` function and comparing them with
    the dummy tokens obtained from the chat data. The function performs various memory management
    operations before returning the final result in
    """
    request_dict = await request.json()
    prompts = request_dict.pop("prompts") 
    # print(prompts, output_tmp)
    outputs = [output.outputs.embedding for output in llm_embeds.embed(prompts)]
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
    uvicorn.run(app, host="0.0.0.0", port=1200)
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