import os
from comfy_api.latest import io # , ui
from ..function import img_process, text_process

self_dir = text_process.self_dir
role_dir = os.path.join(os.path.dirname(self_dir), 'roles')

file_dict = text_process.get_all_file_dict(role_dir, '.txt')

# 加载系统角色，动态选项不大适合，输出时，key 是准的，system_prompt 是没有更新的，除非输入框有编辑
class LoadSystemRole(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LoadSystemRole",
            display_name="Load System Role",
            category="🧪LLMSuite/Shared",
            search_aliases=["VLM", "API", "Role"],
            
            inputs=[
                io.Combo.Input("system_role", options=["none"] + list(file_dict.keys()),),
            ],
            outputs=[
                io.String.Output("system_prompt", ),
            ],
        )
    @classmethod
    def execute(cls, system_role) -> io.NodeOutput:
        if system_role == "none":
            system_prompt = ""
        else:
            file_path = file_dict.get(system_role)
            system_prompt = text_process.get_file_content(file_path) # 到这里才获取，而不是启动获取
        
        return io.NodeOutput(system_prompt,)

class SaveSystemRole(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SaveSystemRole",
            display_name="Save System Role",
            category="🧪LLMSuite/Shared",
            search_aliases=["VLM", "API"],
            is_output_node=True,
            inputs=[
                io.String.Input("role_prompt", force_input=True, optional=False,),
                io.String.Input("role_name", optional=False,),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, role_prompt, role_name) -> io.NodeOutput:
        if not role_name.strip():
            raise ValueError("文件名不能为空！")
        else:
            file_name = text_process.normalize_file_name(role_name, max_length=16)
            file_path = os.path.join(role_dir, f"{file_name}.txt")
            text_process.write_text(file_path, role_prompt)

            global file_dict # 有效，R 刷新后就能看到新列表了
            file_dict = text_process.get_all_file_dict(role_dir, '.txt')
        return io.NodeOutput()

        

NODES = [LoadSystemRole, SaveSystemRole]
