from typing_extensions import override
from comfy_api.latest import ComfyExtension
from .nodes import api_llm, shared #, api_i2i, aaa

# 注册
class LLMSuiteExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [*api_llm.NODES, *shared.NODES, ]

async def comfy_entrypoint():
    return LLMSuiteExtension()

# WEB_DIRECTORY = "./web"
# __all__ = ["WEB_DIRECTORY"]