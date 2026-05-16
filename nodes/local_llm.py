from comfy_api.latest import io
from ..function import text_process

# from comfy_extras.nodes_textgen import TextGenerate
class CLIP2TextGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # Define dynamic combo options for sampling mode
        sampling_options = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input("temperature", default=0.7, min=0.01, max=2.0, step=0.000001),
                    io.Int.Input("top_k", default=64, min=0, max=1000),
                    io.Float.Input("top_p", default=0.95, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01),
                    io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                    io.Float.Input("presence_penalty", optional=True, default=0.0, min=0.0, max=5.0, step=0.01),
                ]
            ),
            io.DynamicCombo.Option(
                key="off",
                inputs=[]
            ),
        ]

        return io.Schema(
            node_id="CLIP2TextGenerate",
            display_name="CLIP To Text Generate",
            category="🧪LLMSuite/Local",
            search_aliases=["LLM", "gemma"],
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("system_prompt", force_input=True, optional=True,),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, default=""),
                io.Image.Input("image", optional=True),
                io.Image.Input("video", optional=True, tooltip="Video frames as image batch. Assumed to be 24 FPS; subsampled to 1 FPS internally."),
                io.Audio.Input("audio", optional=True),
                io.Int.Input("max_length", default=1024, min=1, max=2048),
                io.DynamicCombo.Input("sampling_mode", options=sampling_options, display_name="Sampling Mode"),
                io.Boolean.Input("thinking", optional=True, default=False, tooltip="Operate in thinking mode if the model supports it."),
                io.Boolean.Input("use_default_template", optional=True, default=True, tooltip="Use the built in system prompt/template if the model has one.", advanced=True),
            ],
            outputs=[
                io.String.Output(display_name="generated_text"),
                io.String.Output(display_name="thinking_text"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, max_length, sampling_mode, image=None, thinking=False, use_default_template=True, video=None, audio=None, system_prompt=None) -> io.NodeOutput:
        # 加的
        if system_prompt is None:
            system_prompt = ""
        if image is None:
            formatted_prompt = f"<start_of_turn>system\n{system_prompt.strip()}<end_of_turn>\n<start_of_turn>user\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n<start_of_turn>model\n"
        else:
            formatted_prompt = f"<start_of_turn>system\n{system_prompt.strip()}<end_of_turn>\n<start_of_turn>user\n\n<image_soft_token>\n\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n<start_of_turn>model\n"

        tokens = clip.tokenize(formatted_prompt, image=image, skip_template=not use_default_template, min_length=1, thinking=thinking, video=video, audio=audio)

        # Get sampling parameters from dynamic combo
        do_sample = sampling_mode.get("sampling_mode") == "on"
        temperature = sampling_mode.get("temperature", 1.0)
        top_k = sampling_mode.get("top_k", 50)
        top_p = sampling_mode.get("top_p", 1.0)
        min_p = sampling_mode.get("min_p", 0.0)
        seed = sampling_mode.get("seed", None)
        repetition_penalty = sampling_mode.get("repetition_penalty", 1.0)
        presence_penalty = sampling_mode.get("presence_penalty", 0.0)

        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed
        )

        generated_text = clip.decode(generated_ids)

        thinking, answer = text_process.split_llm_output(generated_text)
            
        return io.NodeOutput(answer, thinking)



NODES = [CLIP2TextGenerate, ]