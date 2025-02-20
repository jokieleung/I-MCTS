from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import requests
import ast
import os
import json
import time
import hashlib

OPENAI_APIKEY="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"


def dump_instances(instances, output_filename):
    with open(output_filename, "a", encoding="utf8") as output:
        for instance in instances:
            output.write(json.dumps(instance, ensure_ascii=False) + "\n")


# 计算字符串的MD5
def calculate_md5_string(input_string):
    # 创建md5对象
    md5 = hashlib.md5()

    # 对字符串进行编码，因为update()方法需要字节类型
    md5.update(input_string.encode('utf-8'))

    # 返回十六进制的MD5值
    return md5.hexdigest()


def aes_encrypt(data, key):
    """aes加密函数，如果data不是16的倍数【加密文本data必须为16的倍数！】，那就补足为16的倍数
    :param key:
    :param data:
    """
    iv = "1234567890123456"
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))  # 设置AES加密模式 此处设置为CBC模式
    block_size = AES.block_size

    # 判断data是不是16的倍数，如果不是用b'\0'补足
    if len(data) % block_size != 0:
        add = block_size - (len(data) % block_size)
    else:
        add = 0
    data = data.encode('utf-8') + b'\0' * add
    encrypted = cipher.encrypt(data)  # aes加密
    result = b2a_hex(encrypted)  # b2a_hex encode  将二进制转换成16进制
    return result.decode('utf-8')


def aes_decode(data, key):
    """aes解密
    :param key:
    :param data:
    """
    iv = '1234567890123456'
    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    result2 = a2b_hex(data)  # 十六进制还原成二进制
    decrypted = cipher.decrypt(result2)
    return decrypted.rstrip(b'\0')  # 解密完成后将加密时添加的多余字符'\0'删除


def request(url, data, aes_encrypt_key = "gs540iivzezmidi3"):
    if aes_encrypt_key is not None:
        data = json.dumps(data)
        data = aes_encrypt(data, key=aes_encrypt_key)
        data = {
            "encryptedParam": data
        }
    data=json.dumps(data)
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, data=data, headers=headers)
        response_js = response.json()
    except JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"response is {response.text}")
        raise Exception(f"调用结果不是json")
    if not response_js['success']:
        raise Exception(f"调用失败,结果为: {response_js}")
    
    return response


def request_chatgpt(prompt, model="gpt-4-turbo", params=None, system=None):
    api_key = OPENAI_APIKEY
    messageKey = "mybk_329340_industry_schema_{}_{}".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) ,calculate_md5_string(prompt))
    if system is None:
        system = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-06\nCurrent date: {}".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))
    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    messages = json.dumps(messages)
    queryConditions = {
        "model": model,
        "api_key": api_key,
        "messages": messages,
        "n":"1",
        # "max_tokens":"150",
        # "temperature":"1",
        # "top_p":"1",
        "outputType":"MQ",
        "mqTag":"EC_datahub_industry_schema",
        "mqTopic":"TP_F_SC_industry_schema",
        "messageKey":messageKey
    }
    if params is not None and isinstance(params, dict):
        queryConditions.update(params)

    content = "请求失败"
    url = 'https://zdfmng.alipay.com/commonQuery/queryData'
    asyn_chatgpt_query_param = {
        "serviceName": "chatgpt_prompts_completions_query_dataview",
        "visitDomain": "BU_it",
        "visitBiz": "BU_it_gpt4",
        "visitBizLine": "BU_it_gpt4_zifu",
        "cacheInterval": -1,
        "queryConditions": queryConditions
    }
    response = request(url, data=asyn_chatgpt_query_param)
    reponse_data = response.json()["data"]["values"]["data"]
    reponse_data = ast.literal_eval("'" + reponse_data + "'")
    reponse_data = reponse_data.replace('&quot;', '"')
    reponse_data = reponse_data.replace("&#39;", "'")
    content = json.loads(reponse_data)
    if 'choices' in content:
        content = content['choices']
        content = content[0]['message']['content']
    return content


def asyn_request_chatgpt(prompt, model="gpt-4o", params=None, system=None):
    api_key = OPENAI_APIKEY
    messageKey = "mybk_329340_industry_schema_{}_{}".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) ,calculate_md5_string(prompt))
    if system is None:
        system = "You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-06\nCurrent date: {}".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))
    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    messages = json.dumps(messages)
    queryConditions = {
        "model": model,
        "api_key": api_key,
        "messages": messages,
        "n":"1",
        # "max_tokens":"150",
        # "temperature":"1",
        # "top_p":"1",
        "outputType":"MQ",
        "mqTag":"EC_datahub_industry_schema",
        "mqTopic":"TP_F_SC_industry_schema",
        "messageKey":messageKey
    }
    if params is not None and isinstance(params, dict):
        queryConditions.update(params)

    content = "请求失败"
    url = 'https://zdfmng.alipay.com/commonQuery/queryData'
    asyn_chatgpt_query_param = {
        "serviceName": "asyn_chatgpt_prompts_completions_query_dataview",
        "visitDomain": "BU_it",
        "visitBiz": "BU_it_gpt4",
        "visitBizLine": "BU_it_gpt4_zifu",
        "cacheInterval": -1,
        "queryConditions": queryConditions
    }
    response = request(url, data=asyn_chatgpt_query_param)
    reponse_data_values = response.json()["data"]["values"]
    reponse_data = reponse_data_values["data"]
    reponse_data = ast.literal_eval("'" + reponse_data + "'")
    reponse_data = reponse_data.replace('&quot;', '"')
    reponse_data = reponse_data.replace("&#39;", "'")
    reponse_data = json.loads(reponse_data)
    reponse_messageKey = reponse_data_values["messageKey"]

    if reponse_data["success"] and reponse_messageKey:
        retry_times = 0
        while True:
            retry_times += 1
            if retry_times > 200:
                break
            time.sleep(3)
            asyn_chatgpt_response_param = {
                "serviceName":"chatgpt_response_query_dataview",
                "visitDomain": "BU_it",
                "visitBiz": "BU_it_gpt4",
                "visitBizLine": "BU_it_gpt4_zifu",
                "cacheInterval": -1,
                "queryConditions":{
                    "messageKey": reponse_messageKey
                }
            }
            response = request(url, data=asyn_chatgpt_response_param)
            content = response.json()["data"]["values"]
            if "response" not in content:
                # print("retry {} chatgpt_response_query_dataview service...".format(retry_times))
                continue
            content = content["response"]
            content = ast.literal_eval("'" + content + "'")
            content = content.replace('&quot;', '"')
            content = content.replace("&#39;", "'")
            content = json.loads(content)
            # if 'choices' in content:
            #     content = content['choices']
            #     content=content[0]['message']['content']
            break
    return content

def my_asyn_request_chatgpt(messages, model="gpt-4o", params=None, system=None):
    api_key = OPENAI_APIKEY
    messageKey = "mybk_329340_industry_schema_{}_{}".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) ,calculate_md5_string(str(messages)))
    
    # messages = [
    #     {
    #         "role": "system",
    #         "content": system
    #     },
    #     {
    #         "role": "user",
    #         "content": prompt
    #     }
    # ]
    messages = json.dumps(messages)
    queryConditions = {
        "model": model,
        "api_key": api_key,
        "messages": messages,
        "n":"1",
        # "max_tokens":"150",
        # "temperature":"1",
        # "top_p":"1",
        "outputType":"MQ",
        "mqTag":"EC_datahub_industry_schema",
        "mqTopic":"TP_F_SC_industry_schema",
        "messageKey":messageKey
    }
    if params is not None and isinstance(params, dict):
        queryConditions.update(params)

    content = "请求失败"
    url = 'https://zdfmng.alipay.com/commonQuery/queryData'
    asyn_chatgpt_query_param = {
        "serviceName": "asyn_chatgpt_prompts_completions_query_dataview",
        "visitDomain": "BU_it",
        "visitBiz": "BU_it_gpt4",
        "visitBizLine": "BU_it_gpt4_zifu",
        "cacheInterval": -1,
        "queryConditions": queryConditions
    }
    response = request(url, data=asyn_chatgpt_query_param)
    reponse_data_values = response.json()["data"]["values"]
    reponse_data = reponse_data_values["data"]
    reponse_data = ast.literal_eval("'" + reponse_data + "'")
    reponse_data = reponse_data.replace('&quot;', '"')
    reponse_data = reponse_data.replace("&#39;", "'")
    reponse_data = json.loads(reponse_data)
    reponse_messageKey = reponse_data_values["messageKey"]

    if reponse_data["success"] and reponse_messageKey:
        retry_times = 0
        while True:
            retry_times += 1
            if retry_times > 200:
                break
            time.sleep(3)
            asyn_chatgpt_response_param = {
                "serviceName":"chatgpt_response_query_dataview",
                "visitDomain": "BU_it",
                "visitBiz": "BU_it_gpt4",
                "visitBizLine": "BU_it_gpt4_zifu",
                "cacheInterval": -1,
                "queryConditions":{
                    "messageKey": reponse_messageKey
                }
            }
            response = request(url, data=asyn_chatgpt_response_param)
            content = response.json()["data"]["values"]
            if "response" not in content:
                # print("retry {} chatgpt_response_query_dataview service...".format(retry_times))
                continue
            content = content["response"]
            content = ast.literal_eval("'" + content + "'")
            content = content.replace('&quot;', '"')
            content = content.replace("&#39;", "'")
            content = content.replace("&gt;", ">")
            content = content.replace("&lt;", "<")
            content = content.replace("&amp;", "&")
            content = json.loads(content)
            # if 'choices' in content:
            #     content = content['choices']
            #     content=content[0]['message']['content']
            break
    return content


def chatgpt_api(prompt, model="gpt-4-turbo", params=None, asyn_request=False, logging=False, system=None, trace_task=None):
    try:
        if not asyn_request:
            response = request_chatgpt(prompt, model, params, system)
        else:
            response = asyn_request_chatgpt(prompt, model, params, system)
    except Exception as e:
        print("[retry] request asyn_request_chatgpt()...", e)
        response = asyn_request_chatgpt(prompt, model, params)
    if logging:
        instances = [{"input": prompt, "output": response, "model": model, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]
        if system is not None:
            instances[0]["system"] = system
        if params is not None:
            instances[0]["params"] = params
        if trace_task is not None:
            instances[0]["trace_task"] = trace_task
        os.makedirs(os.path.join(os.path.dirname(__file__), "log"), exist_ok=True)
        output_filename = os.path.join(os.path.dirname(__file__), "log", "history_model_{}.jsonl".format(time.strftime("%Y-%m-%d", time.localtime())))
        dump_instances(instances, output_filename=output_filename)
    return response


