from openai import OpenAI
import os
import sys
sys.path.append(os.path.dirname(__file__))
# print("sys.path:", sys.path)
from asyn_chatgpt_api import asyn_request_chatgpt
import asyncio
import openai
import argparse
import os 
openai.base_url = "http://industrial.models.antcloud.mybank-inc.cn/v1/"
openai.api_key = "none"

def askLLamaFactory(query, model="sft",logprobs=False, temperature=0.4):
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(
            os.environ.get("API_PORT", 8887)),
    )
    messages = [{"role": "user", "content": query}]
    result = client.chat.completions.create(
        messages=messages, model=model, temperature=temperature, max_tokens=16384
        ,logprobs=logprobs
        # ,do_sample =True
        , top_p=5
        )
    # response = result.choices[0].message.content
    content = result.choices[0].message.content
    if logprobs:
      logprobs = result.choices[0].logprobs.content
    else:
      logprobs = None
    return (content, logprobs)
    

def askg4(prompt, model="gpt-4o",logprobs=False, temperature=0.4):
    ret = asyn_request_chatgpt(prompt, model=model, params={"temperature": temperature, "logprobs": logprobs}, system=None)
    content = ret['choices'][0]['message']['content']
    if logprobs:
      logprobs = ret['choices'][0]['logprobs']['content']
    else:
      logprobs = None
    return (content, logprobs)

# Only support ['mybkllm', 'qwen2-72b-x', 'qwen2dot5-72b', 'qwen2dot5-72b-7b', 'qwen2dot5-14b', 'qwen2-72b', 'qwen2vl', 'docwen', 'mybkllmv4', 'qwen2dot5-72b-fp8', 'qwen2.5-7b', 'qwen2dot5-7b', 'qwen2dot5-7b:diwen_all_sft_lora_v4', 'qwen2dot5-7b:lora-script', 'qwen2dot5-7b:med_ee', 'qwen2dot5-32b', 'qwen15-32b', 'qwen2dot5-7b-vllm-l20-py', 'qwen2dot5-7b-vllm-l20-py:text', 'minicpmv-v2dot6', 'glm4-9b', 'qwen2dot5-coder-7b', 'qwen2dot5-7b-fp8-sg', 'qwen2.5-fp8']"}
def askqwen(query, model="qwen2-72b",logprobs=False, temperature=0.4):
    messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": f"{query}"},
    ]
    response = openai.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    stream=False,
    max_tokens=16384
    ,logprobs=logprobs
    # ,top_logprobs=1
    )

    content = response.choices[0].message.content
    if logprobs:
      logprobs = response.choices[0].logprobs.content
    else:
      logprobs = None
    return (content, logprobs)
    
    # return response.choices[0].message.content    

async def askllm(query, model="qwen2dot5-72b",logprobs=False,temperature=0.4):
  if "sft" in model:
    return askLLamaFactory(query, model,logprobs, temperature)
  elif "gpt" in model:
    return askg4(query, model, logprobs, temperature)
  elif "qwen" in model:
    return askqwen(query, model, logprobs, temperature)



async def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='示例脚本，展示如何设置参数默认值')

    # 添加参数
    parser.add_argument('--prompt', type=str, default='Hello, who are you?', help='提示词')
    parser.add_argument('--model', type=str, default="qwen2-72b", help='模型名称')
    parser.add_argument('--logprobs', action='store_true', default=False, help='是否输出词概率')
    parser.add_argument('--temperature', type=float, default=0.4, help='温度')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    print(args)
    r = await askllm(args.prompt, args.model, args.logprobs, args.temperature)
    print("-----")
    print(r)

if __name__ == '__main__':
    # main()    
    asyncio.run(main())
