from comfy_api.latest import io #, ui
import os
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from ..function import img_process, text_process

api_model_config = text_process.api_model_config

# 仅得到模型选项
def get_model_options(model_dict):
    model_options = []
    for model_name, model_info in model_dict.items():
        option_list = []
        for option_name, value in model_info.items():
            if option_name == "thinking":
                option_list.append(io.Combo.Input("thinking", options=value))
            elif option_name == "max_tokens":
                option_list.append(io.Int.Input("max_tokens", default=value, min=32, max=409600, step=8, control_after_generate=False))
            elif option_name == "temperature":
                option_list.append(io.Float.Input("temperature", default=value, min=0.01, max=2.0, step=0.0001))
            elif option_name == "seed":
                option_list.append(io.Int.Input("seed", default=value, min=0, max=0xffffffffffffffff, control_after_generate=False))
            elif option_name == "web_search":
                option_list.append(io.Boolean.Input("web_search", optional=True, default=value, tooltip="Use the web search function to get better response results.", advanced=True))

                
        model_option_group = io.DynamicCombo.Option(
            key=model_name,
            inputs=option_list,
        )
        model_options.append(model_option_group)
    return model_options

# 从字典获取动态选项，所有平台的
def get_llm_ui_options(api_model_config, platform_list, model_type):
    platform_options = []
    for platform_name, platform_map in api_model_config.items():
        if platform_name in platform_list:
            model_map = platform_map.get("Model_Map")
            model_dict = model_map.get(model_type, None)
            if model_dict is not None:
                model_options = get_model_options(model_dict)
                model_name_group = [io.DynamicCombo.Input("model", options=model_options)]
                platform_group = io.DynamicCombo.Option(key=platform_name, inputs=model_name_group)
                platform_options.append(platform_group)

    return platform_options

# 可以显示所有选项，也是所有 API 模型节点的基础，但不显示此节点
class API_AnyVLM(io.ComfyNode):
    platform_list = ["AutoDL", "VolcEngine"] #  "VolcEngine"
    platform_tip = ' or '.join(platform_list)
    model_type = "VLM"

    @classmethod
    def define_schema(cls) -> io.Schema:
        platform_options = get_llm_ui_options(api_model_config, cls.platform_list, cls.model_type)
        
        return io.Schema(
            node_id="API_AnyVLM",
            display_name="API Any VLM",
            category="🧪LLMSuite/API",
            search_aliases=["VLM", "API"],
            inputs=[
                io.String.Input("system_prompt", force_input=True, optional=True,),
                io.Image.Input("images", optional=True,),
                io.String.Input("user_prompt", multiline=True, default="",),
                io.String.Input("api_token", default="", display_name=f"{cls.platform_tip} api_token", tooltip="Please enter the API Token of the current platform"),
                io.DynamicCombo.Input("platform", options=platform_options, ),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.Custom("INFO").Output(display_name="all_info"),
            ],
        )

    @classmethod
    async def execute(cls, user_prompt, api_token, platform, system_prompt=None, images=None) -> io.NodeOutput:
        platform_name = platform.get("platform")
        if not api_token.strip():
            raise ValueError(f"❌ Please fill in the {platform_name} API Token")
        if system_prompt is not None:
            system_prompt = system_prompt.strip()
        timeout = 180.0

        must_thinking_models = api_model_config[platform_name].get("Must_Thinking_Models", [])
        
        
        model_group = platform.get("model")
        model_name = model_group.get("model")
        thinking = model_group.get("thinking")
        max_tokens = model_group.get("max_tokens", 2048)
        temperature = model_group.get("temperature", 0.7)
        seed = model_group.get("seed", 0)
        web_search = model_group.get("web_search", False)
        
        sdk_name = api_model_config[platform_name]["Model_Map"][cls.model_type][model_name].get("sdk_name", "OpenAI") # "OpenAI"兜底
        
        # opus 之外的模型，Anthropic 无论如何都会关闭思考模式，利用这一点，来关闭 openai 无法关闭思考模式的模型
        if model_name in must_thinking_models and thinking == "none":
            sdk_name = "Anthropic"
            thinking = "none"
            
        base_url = api_model_config[platform_name]["SDK_Map"].get(sdk_name)
        generated_text, response = "", ""
        
        try:
            if sdk_name == "Anthropic":
                messages = text_process.anthropic_prompt(user_prompt, system_prompt, images, messages=None)
                client = AsyncAnthropic(base_url=base_url, api_key=api_token, timeout=timeout,)

                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                
                if system_prompt:
                    kwargs["system"] = system_prompt
                
                if thinking == "none":
                    kwargs["thinking"] = {"type": "disabled"}
                    kwargs["output_config"] = {"effort": "low"}
                else:
                    kwargs["thinking"] = {"type": "adaptive"}
                    kwargs["output_config"] = {"effort": thinking}
    
                response = await asyncio.wait_for(
                    client.messages.create(**kwargs),
                    timeout=timeout
                )
                
                for block in response.content:
                    if block.type == "text":
                        generated_text = block.text
                        break
                    
            elif sdk_name == "OpenAI":
                messages = text_process.openai_prompt(user_prompt, system_prompt, images, messages=None)
                
                client = AsyncOpenAI(base_url=base_url, api_key=api_token, timeout=timeout,)

                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "reasoning_effort": thinking,
                    "seed": seed,
                }
                
                response = await asyncio.wait_for(
                    client.chat.completions.create(**kwargs),
                    timeout=timeout
                )

                generated_text = response.choices[0].message.content

            elif sdk_name == "OpenAI_Responses":
                messages = text_process.openai_responses_prompt(user_prompt, system_prompt, images, messages=None)

                kwargs = {
                    "model": model_name,
                    "input": messages,
                    "max_output_tokens": max_tokens,
                }

                if system_prompt:
                    kwargs["instructions"] = system_prompt
                if web_search:
                    kwargs["tools"] = [{"type": "web_search"}]
                
                if model_name.startswith("gpt"):
                    kwargs["reasoning"] = {"effort": f"{thinking}", "generate_summary": "detailed", "summary": 'detailed'}
                else:
                    # 只根据火山引擎
                    if thinking == "none":
                        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                    else:
                        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                        kwargs["reasoning"] = {"effort": f"{thinking}"}
                        
                    kwargs["temperature"] = temperature
                    

                client = AsyncOpenAI(base_url=base_url, api_key=api_token, timeout=timeout,)
                response = await asyncio.wait_for(
                    client.responses.create(**kwargs),
                    timeout=timeout
                )
                
                for block in response.output:
                    if block.type == "message":
                        for c in block.content:
                            if c.type == "output_text":
                                generated_text = c.text
                                break
      
            else:
                raise ValueError(f"❌ Unknown SDK name: {sdk_name}")
                
        except Exception as e:
            text_process.exception_process(e, platform_name)
            
        all_info = (sdk_name, response)
        return io.NodeOutput(generated_text, all_info)
        

# AutoDL VLM 专用
class API_AutoDLVLM(API_AnyVLM):
    platform_list = ["AutoDL",] #  "VolcEngine"
    platform_tip = ' or '.join(platform_list)
    model_type = "VLM"

    @classmethod
    def define_schema(cls) -> io.Schema:
        platform_options = get_llm_ui_options(api_model_config, cls.platform_list, cls.model_type)
        parent_schema = super().define_schema()
        
        return io.Schema(
            node_id="API_AutoDLVLM",
            display_name="API AutoDL VLM",
            category="🧪LLMSuite/API",
            search_aliases=["VLM", "API"],
            inputs=parent_schema.inputs,
            outputs=parent_schema.outputs,
        )



# AutoDL LLM 专用，没有图像输入，不应该使用 super()
class API_AutoDLLLM(API_AnyVLM):
    platform_list = ["AutoDL", ]
    platform_tip = ' or '.join(platform_list)
    model_type = "LLM"

    @classmethod
    def define_schema(cls) -> io.Schema:
        platform_options = get_llm_ui_options(api_model_config, cls.platform_list, cls.model_type)
        
        return io.Schema(
            node_id="API_AutoDLLLM",
            display_name="API AutoDL LLM",
            category="🧪LLMSuite/API",
            search_aliases=["LLM", "API"],
            inputs=[
                io.String.Input("system_prompt", force_input=True, optional=True,),
                io.String.Input("user_prompt", multiline=True, default="",),
                io.String.Input("api_token", default="", display_name=f"{cls.platform_tip} api_token", tooltip="Please enter the API Token of the current platform"),
                io.DynamicCombo.Input("platform", options=platform_options, ),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.Custom("INFO").Output(display_name="all_info"),
            ],
        )

# 火山引擎 VLM 专用
class API_VolcEngineVLM(API_AnyVLM):
    platform_list = ["VolcEngine", ]
    platform_tip = ' or '.join(platform_list)
    model_type = "VLM" # 必须确保存在

    @classmethod
    def define_schema(cls) -> io.Schema:
        platform_options = get_llm_ui_options(api_model_config, cls.platform_list, cls.model_type)
        parent_schema = super().define_schema()
        
        return io.Schema(
            node_id="API_VolcEngineVLM",
            display_name="API VolcEngine VLM",
            category="🧪LLMSuite/API",
            search_aliases=["VLM", "API"],
            inputs=parent_schema.inputs,
            outputs=parent_schema.outputs,
        )


    
# 反馈信息分析
class API_InfoAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="API_InfoAnalysis",
            display_name="API Model Info Analysis",
            category="🧪LLMSuite/API",
            search_aliases=["VLM", "API"],
            inputs=[
                io.Custom("INFO").Input("all_info", ),
            ],
            outputs=[
                io.String.Output(display_name="request_id"),
                io.String.Output(display_name="thinking"),
                io.String.Output(display_name="token_statistics"),
            ],
        )
    @classmethod
    def execute(cls, all_info) -> io.NodeOutput:
        sdk_name, response = all_info
        parsed = text_process.get_api_parse(sdk_name, response)

        token_statistics = (
            f"input_tokens: {parsed['usage']['input_tokens']}\n"
            f"output_tokens: {parsed['usage']['output_tokens']}\n"
            f" - thinking_tokens: {parsed['usage']['reasoning_tokens']}\n"
            f" - visible_tokens: {parsed['usage']['visible_tokens']}\n\n"
            f"total_tokens: {parsed['usage']['total_tokens']}"
        )
        return io.NodeOutput(parsed['request_id'], parsed['reasoning'], token_statistics)

        
        
NODES = [API_AutoDLVLM, API_AutoDLLLM, API_VolcEngineVLM, API_InfoAnalysis,]  # API_AnyVLM 最好不要显示


