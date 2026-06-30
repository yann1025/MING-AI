# -*- coding: utf-8 -*-
"""
MING 医疗对话 - 单文件启动脚本（Gradio 4.x 版）
基座: Qwen1.5-1.8B-Chat  +  LoRA: MING-MOE-1.8B
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import torch
import gradio as gr
from threading import Thread
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ========== 路径配置 ==========
BASE_MODEL_PATH = r"D:\code\medicine-MING\medicine-MING\models\qwen\Qwen1.5-1.8B-Chat"
LORA_PATH       = r"D:\code\medicine-MING\medicine-MING\models\MING-MOE-1.8B"

USE_4BIT = False

# ========== 加载模型 ==========
print("🔄 正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("🔄 正在加载基座模型 Qwen1.5-1.8B-Chat ...")
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

print(f"✅ 基座加载完成，显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

print("🔄 正在加载医疗 LoRA (MING-MOE-1.8B) ...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("✅ 模型加载完成！")

# ========== 推理函数（流式） ==========
SYSTEM_PROMPT = "你是一位专业、严谨、温暖的中文医学助手 MING。请基于医学知识为用户提供准确、可靠的健康建议，必要时提醒用户就医。"

def chat_stream(message, history):
    """ history: [[user, bot], [user, bot], ...] """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for new_text in streamer:
        partial += new_text
        yield partial

# ========== Gradio 4.x 界面 ==========
with gr.Blocks(title="MING 医疗助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 MING 医疗助手  
        基于 **Qwen1.5-1.8B-Chat + MING-MOE 医疗 LoRA**  
        ⚠️ 本助手仅供学习参考，不能替代专业医生诊断。
        """
    )
    chatbot = gr.Chatbot(height=500, label="对话窗口")
    msg = gr.Textbox(label="请输入您的健康问题", placeholder="例如：最近总是失眠该怎么办？")
    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话")

    def user_submit(user_message, history):
        return "", history + [[user_message, ""]]

    def bot_respond(history):
        user_message = history[-1][0]
        history[-1][1] = ""
        for partial in chat_stream(user_message, history[:-1]):
            history[-1][1] = partial
            yield history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_respond, chatbot, chatbot
    )
    clear_btn.click(lambda: None, None, chatbot, queue=False)

    gr.Examples(
        examples=[
            "我最近总是头晕，可能是什么原因？",
            "高血压患者饮食上要注意什么？",
            "感冒和流感怎么区分？",
            "糖尿病前期应该怎样干预？",
        ],
        inputs=msg,
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)