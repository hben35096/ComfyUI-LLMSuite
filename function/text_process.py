import yaml
import os
import unicodedata
import re
from ..function import img_process

self_dir = os.path.abspath(os.path.dirname(__file__))
config_dir = os.path.join(os.path.dirname(self_dir), "config_files")

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

api_model_config = load_config(os.path.join(config_dir, "api_model_config.yaml"))


# my_input=[
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "这张图里有什么？"},
#             {"type": "image_url", "image_url": "https://codewithgpu-image-1310972338.cos.ap-beijing.myqcloud.com/80455-894512147-vvEiRXNzIG41q72EkiFT.jpg"}
#         ]
#     }
# ]

# 发送信息处理 OpenAI
def openai_prompt(user_prompt, system_prompt=None, images=None, messages=None):
    user_prompt = user_prompt.strip()
    if messages is None:
        messages = []
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    
    user_content_list = []
    if images is not None:
        for img in images:
            base64_encoded, mime_type = img_process.tensor2base64(img, "JPEG")
            if base64_encoded is not None:
                base64_str = f"data:{mime_type};base64,{base64_encoded}" # 拼接data URI格式（供API调用）
                img_dist =  {"type": "image_url", "image_url": {"url": base64_str}}
                user_content_list.append(img_dist)
    if user_prompt:
        user_content_list.append({"type": "text", "text": user_prompt}) # 统一放图像后面吧
        
    user_msgs = {"role": "user", "content": user_content_list} # 内容列表可以是空列表
    messages.append(user_msgs)
    return messages

# my_input=[
#     {
#         "role": "user",
#         "content": [
#             {"type": "input_text", "text": "这张图里有什么？"},
#             {"type": "input_image", "image_url": "https://codewithgpu-image-1310972338.cos.ap-beijing.myqcloud.com/80455-894512147-vvEiRXNzIG41q72EkiFT.jpg"}
#         ]
#     }
# ]
    
# 发送信息处理 OpenAI Responses，豆包要求至少有一项输入，GPT则无所谓，不确定要这里判断还是
def openai_responses_prompt(user_prompt, system_prompt=None, images=None, messages=None):
    user_prompt = user_prompt.strip()
    if messages is None:
        messages = []
    user_content_list = []
    if images is not None:
        for img in images:
            base64_encoded, mime_type = img_process.tensor2base64(img, "JPEG")
            if base64_encoded is not None:
                base64_str = f"data:{mime_type};base64,{base64_encoded}" # 拼接data URI格式（供API调用）
                img_dist =  {"type": "input_image", "image_url": base64_str}
                user_content_list.append(img_dist)
                
    if user_prompt:
        user_content_list.append({"type": "input_text", "text": user_prompt})
    if not user_content_list:
        raise ValueError("❌ At least one input is required, either 'user_prompt' or 'images'")
        
    user_msgs = {"role": "user", "content": user_content_list}
    messages.append(user_msgs)
    return messages

# messages=[
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Image 1:"},
#             {"type": "image", "source": {"type": "base64", "media_type": image1_media_type, "data": image1_data,},},
#             {"type": "text", "text": "Image 2:"},
#             {"type": "image","source": {"type": "base64","media_type": image2_media_type,"data": image2_data,},},
#             {"type": "text", "text": "How are these images different?"},
#         ],
#     }
# ]

# 发送信息处理 Anthropic
def anthropic_prompt(user_prompt, system_prompt=None, images=None, messages=None):
    user_prompt = user_prompt.strip()
    if messages is None:
        messages = []
    
    user_content_list = []
    if images is not None:
        number = 0
        for img in images:
            base64_encoded, mime_type = img_process.tensor2base64(img, "JPEG")
            if base64_encoded is not None:
                number += 1
                content_text = {"type": "text", "text": f"{number}:"}
                content_image = {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": base64_encoded,}}
                user_content_list.extend([content_text, content_image])
    if user_prompt:
        content_user = {"type": "text", "text": user_prompt}
        user_content_list.append(content_user)

    if not system_prompt and not user_content_list:
        raise ValueError("❌ At least one input is required, either 'system_prompt' or 'user_prompt")
    user_msgs = {"role": "user", "content": user_content_list} # 目前测试的模型都允许用户内容为空列表，但系统提示和用户内容至少要有一个，上面已做判断
    messages.append(user_msgs)

    return messages


# 1. 预设默认返回值（确保任何情况下都有完整结构）
def result_empty_dict():
    result = {
        "model": "",
        "request_id": "",
        "finish_reason": "",
        "reply": "",
        "reasoning": "",
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "visible_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "text_tokens": 0
        }
    }
    return result

# 返回信息解析
def openai_parse_response(response):
    result = result_empty_dict()
    
    if not response:
        return result

    try:
        reasoning_tokens = 0
        result["model"] = getattr(response, "model", "") or ""
        result["request_id"] = getattr(response, "id", "") or ""

        choices = getattr(response, "choices", None) or []
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            choice = choices[0]
            result["finish_reason"] = getattr(choice, "finish_reason", "") or ""

            message = getattr(choice, "message", None)
            if message:
                result["reply"] = getattr(message, "content", "") or ""
                result["reasoning"] = getattr(message, "reasoning_content", "") or ""

        # 使用情况
        usage = getattr(response, "usage", None)
        if usage:
            p_tok = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
            c_tok = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
            c_tok = c_tok if isinstance(c_tok, (int, float)) else 0 # 确保是数值
            
            result["usage"]["input_tokens"] = p_tok or 0
            result["usage"]["output_tokens"] = c_tok or 0
            result["usage"]["total_tokens"] = getattr(usage, "total_tokens", 0) or 0
            
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                reasoning_tokens = reasoning_tokens if isinstance(reasoning_tokens, (int, float)) else 0 # 确保是数值
                result["usage"]["reasoning_tokens"] = reasoning_tokens
                result["usage"]["text_tokens"] = getattr(details, "text_tokens", 0) or 0
                
            if reasoning_tokens == 0:
                result["usage"]["visible_tokens"] = c_tok
            else:
                result["usage"]["visible_tokens"] = c_tok - reasoning_tokens
            

    except Exception:
        pass

    return result


def anthropic_parse_response(response):
    result = result_empty_dict()
    
    if not response:
        return result

    try:
        result["model"] = getattr(response, "model", "") or ""
        result["request_id"] = getattr(response, "id", "") or ""
        result["finish_reason"] = getattr(response, "stop_reason", "") or ""
        contents = getattr(response, "content", None) or []

        if isinstance(contents, (list, tuple)) and len(contents) > 0:
            for block in contents:
                if block.type == "text":
                    result["reply"] = getattr(block, "text", None)
                if block.type == 'thinking':
                    result["reasoning"] = getattr(block, "thinking", None)

        # Token 统计
        usage = getattr(response, "usage", None)
        if usage:
            p_tok = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
            c_tok = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)

            p_tok = p_tok if isinstance(p_tok, (int, float)) else 0
            c_tok = c_tok if isinstance(c_tok, (int, float)) else 0
            
            result["usage"]["input_tokens"] = p_tok
            result["usage"]["output_tokens"] = c_tok
            
            result["usage"]["total_tokens"] = p_tok + c_tok
            result["usage"]["visible_tokens"] = c_tok - 0

    except Exception:
        pass

    return result

def openai_responses_parse_response(response):
    result = result_empty_dict()
    
    if not response:
        return result

    try:
        reasoning_tokens = 0
        result["model"] = getattr(response, "model", "") or ""
        result["request_id"] = getattr(response, "id", "") or ""

        outputs = getattr(response, "output", None) or []
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            for block in outputs:
                if block.type == "message":
                    for c in block.content:
                        if c.type == "output_text":
                            result["reply"] = c.text
                if block.type == "reasoning":
                    summary = getattr(block, "summary", None) or []
                    if isinstance(summary, (list, tuple)) and len(outputs) > 0:
                        for c in summary:
                            if c.type == "summary_text":
                                result["reasoning"] = c.text

        # Token 统计
        usage = getattr(response, "usage", None)
        if usage:
            p_tok = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
            p_tok = p_tok if isinstance(p_tok, (int, float)) else 0
            
            c_tok = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
            c_tok = c_tok if isinstance(c_tok, (int, float)) else 0 # 确保是数值

            result["usage"]["input_tokens"] = p_tok or 0
            result["usage"]["output_tokens"] = c_tok or 0
            
            total_tokens = getattr(usage, "total_tokens", 0)
            result["usage"]["total_tokens"] = total_tokens if isinstance(total_tokens, (int, float)) and total_tokens > 0 else (p_tok + c_tok)

            # 推理 tokens 细节
            details = getattr(usage, "output_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
                reasoning_tokens = reasoning_tokens if isinstance(reasoning_tokens, (int, float)) else 0 # 确保是数值
                result["usage"]["reasoning_tokens"] = reasoning_tokens
                result["usage"]["text_tokens"] = getattr(details, "text_tokens", 0) or 0
                
            if reasoning_tokens == 0:
                result["usage"]["visible_tokens"] = c_tok
            else:
                result["usage"]["visible_tokens"] = c_tok - reasoning_tokens
    except Exception:
        pass

    return result
    
# 区分并获取统计信息
def get_api_parse(sdk_name, response):
    if sdk_name == "OpenAI":
        parse_dict = openai_parse_response(response)
    elif sdk_name == "Anthropic":
        parse_dict = anthropic_parse_response(response)
    elif sdk_name == "OpenAI_Responses":
        parse_dict = openai_responses_parse_response(response)
    else:
        parse_dict = result_empty_dict()
    return parse_dict

# 获取目录下所有非隐藏文件
def get_all_file_dict(dir_path, exp_name):
    file_dict = {}
    for name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, name)
        if os.path.isfile(file_path):
            if not name.startswith('.') and name.endswith(exp_name):
                base_name = name.replace(exp_name, "")
                file_dict[base_name] = file_path
    # 按创建时间降序排序（最新的文件在前）
    # file_path_list = sorted(file_path_list, key=lambda x: os.path.getctime(x), reverse=True)
    
    return file_dict

# 读取文件内容
def get_file_content(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            file_content = ''
    else:
        file_content = ''
    return file_content.strip()

# 这个没用到
def get_name_and_content(dir_path, exp_name):
    file_dict = get_all_file_dict(dir_path, exp_name)
    content_dict = {"none": ""}
    for filename, file_path in file_dict.items():
        name = filename.replace(exp_name, "")
        content = get_file_content(file_path)
        content_dict[name] = content
    return content_dict


# 报错处理
# import openai
# import anthropic

def exception_process(e, platform_name=None):
    ERROR_MAP = {
        "AuthenticationError": {"type": ValueError, "message": f"🔐 Authentication Failed: API Key is invalid, incorrect, or not filled in, Are you using the API Token of {platform_name}?"},
        "PermissionDeniedError": {"type": PermissionError, "message": "🚫 No permission to access this model"},
        "BadRequestError": {"type": ValueError, "message": "⚠️ Invalid request parameters"},
        "NotFoundError": {"type": LookupError, "message": "🔍 The requested resource does not exist: The model name or interface path may be incorrect"},
        "UnprocessableEntityError": {"type": ValueError, "message": "🧾 The request cannot be processed (token limit exceeded or logical conflict)"},
        "APITimeoutError": {"type": TimeoutError, "message": "⌛ Request Timeout"},
        "RateLimitError": {"type": ConnectionError, "message": "⏳ Call frequency exceeded, please try again later"},
        "APIStatusError": {"type": ConnectionError, "message": "❗ The service returns an abnormal status code (non-2xx), which may be a temporary issue or an unknown error."},
        "InternalServerError": {"type": ConnectionError, "message": "🔥 Internal Server Error, please try again later"},
        "APIConnectionError": {"type": ConnectionError, "message": "🌐 Network connection failed, unable to connect to the server"},
        "APIError": {"type": RuntimeError, "message": "🔥 Server Error"},
        "TimeoutError": {"type": TimeoutError, "message": "❌ Request timeout"},
    }
    
    ADD_MSG_MAP = {
        "Insufficient account balance": "💰 Insufficient balance",
    }
    
    err_text = str(e)
    add_msg_list = []
    for key, value in ADD_MSG_MAP.items():
        if key in err_text:
            add_msg_list.append(value)
    add_msg_str = "\n".join(add_msg_list)

    
    name = type(e).__name__
    rule = ERROR_MAP.get(name)

    if rule:
        msg = rule["message"]
        exc_type = rule["type"]

        if "{status_code}" in msg:
            status = getattr(e, "status_code", None)
            msg = msg.format(status_code=status)

        raise exc_type(f"{msg}\n{add_msg_str}") from e

    raise e

# 文件名规范化
def normalize_file_name(name: str, max_length: int = 16) -> str:
    name = name.strip()
    name = unicodedata.normalize('NFKD', name) # 转换 Unicode 全角字符到半角（如中文标点）
    name = re.sub(r'[\s\-–—]+', '_', name)  # 空格和各种横线变成下划线
    # 移除非法字符（保留字母、数字、下划线）
    name = re.sub(r'[^\w]', '', name)  # \w = [a-zA-Z0-9_]
    # 截断过长的名称
    return name[:max_length]

# 写入文件内容
def write_text(file_path, text):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"The content has been successfully written to: {file_path}")
    except Exception as e:
        raise ValueError("File write failed:", e)