# %% [markdown]
# Source notebook: Image_Reasoning_and_Grounding.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L3: Image Reasoning and Grounding

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load API keys

# %%
# Notebook cell 5
import os
from utils import get_llama_api_key, get_llama_base_url, get_together_api_key

llama_api_key = get_llama_api_key()
llama_base_url = get_llama_base_url()
together_api_key = get_together_api_key()

# %% [markdown]
# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# %% [markdown]
# ## Llama helpers

# %%
# Notebook cell 8
from utils import llama4, llama4_together

# %%
# Notebook cell 9
import base64

def encode_image_to_base64(image_path):
    """
    Encode a local image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# %% [markdown]
# ## Image Grounding

# %%
# Notebook cell 11
from utils import display_local_image
display_local_image("images/tools.png")

# %%
# Notebook cell 12
prompt = """Which tools in the image can be used for measuring length?
Provide bounding boxes for every recognized item."""

# %%
# Notebook cell 13
base64_tools = encode_image_to_base64("images/tools.png")

print(llama4(prompt,[f"data:image/jpeg;base64,{base64_tools}"]))

# %%
# Notebook cell 14
from utils import parse_output, draw_bounding_boxes
output = llama4(prompt, [f"data:image/jpeg;base64,{base64_tools}"])
tools = parse_output(output)
draw_bounding_boxes("images/tools.png", tools)

# %% [markdown]
# ## Analyze table in PDF

# %%
# Notebook cell 16
from pathlib import Path
from pypdf import PdfReader

def pdf2text(file : str):
  text = ''
  with Path(file).open("rb") as f:
    reader = PdfReader(f)
    text = "\n\n".join([page.extract_text() for page in reader.pages])

  return text

# %%
# Notebook cell 17
meta_q4_2024_txt = pdf2text("Meta-Reports-Fourth-Quarter-and-Full-Year-2024-Results-2025.pdf")

# %%
# Notebook cell 18
start = meta_q4_2024_txt.find("Fourth Quarter and Full Year 2024 Financial Highlights")
print(meta_q4_2024_txt[start:start+1000])

# %%
# Notebook cell 19
prompt = f"""How much is 2024 operating margin based on Meta's financial
quarter report below:
{meta_q4_2024_txt}
"""
print(llama4(prompt))

# %%
# Notebook cell 20
display_local_image("images/meta-q4-2024-highlights.png")

# %%
# Notebook cell 21
base64_meta = encode_image_to_base64("images/meta-q4-2024-highlights.png")
prompt = """How much is 2024 operating margin based on Meta's financial
report?"""
print(llama4(prompt, [f"data:image/jpeg;base64,{base64_meta}"]))

# %% [markdown]
# ## Generating code from a screenshot

# %%
# Notebook cell 23
vid_frame_address = "images/video_frame_1440.jpg"
display_local_image(vid_frame_address)

# %%
# Notebook cell 24
base64_image = encode_image_to_base64(vid_frame_address)
prompt = """If I want to change the temperature on the image,
where should I click? Return the bounding box for the location."""
output = llama4(prompt, [f"data:image/jpeg;base64,{base64_image}"])
output

# %%
# Notebook cell 25
prompt = """"Write a python script that uses Gradio to implement
the chatbot UI in the image."""

output = llama4(prompt,[f"data:image/jpeg;base64,{base64_image}"],
                model="Llama-4-Maverick-17B-128E-Instruct-FP8")
print(output)

# %%
# Notebook cell 26
import gradio as gr

def chatbot_settings(temp, max_tokens, repetition, top_p, streaming):
    return f"Temperature: {temp}, Max Tokens: {max_tokens}, Repetition: {repetition}, Top P: {top_p}, Streaming: {streaming}"

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Llama-4-Maverick-17B-128E-Instruct-FP8")
                msg = gr.Textbox(label="Enter message...")
            with gr.Column(scale=1):
                with gr.Group("Model settings"):
                    temp = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.6)
                    max_tokens = gr.Slider(label="Max tokens", minimum=0, maximum=4096, value=2048)
                    repetition = gr.Slider(label="Repetition", minimum=0, maximum=2, value=1.0)
                    top_p = gr.Slider(label="Top P", minimum=0, maximum=1, value=0.9)
                    streaming = gr.Checkbox(label="Streaming", value=True)
                    advanced = gr.Button("Advanced")
                    json_schema = gr.Button("JSON schema")
                    tools = gr.Button("Tools")

        def respond(message, history, temp, max_tokens, repetition, top_p, streaming):
            # Here you would implement your chatbot's response logic
            # For now, it just echoes the input
            bot_message = f"Echo: {message}"
            history.append((message, bot_message))
            return "", history

        msg.submit(respond, [msg, chatbot, temp, max_tokens, repetition, top_p, streaming], [msg, chatbot])

    demo.launch()

if __name__ == "__main__":
    main()

# %% [markdown]
# ## Solving Math

# %%
# Notebook cell 28
display_local_image("images/simple_math.png")

# %%
# Notebook cell 29
base64_math = encode_image_to_base64("images/simple_math.png")
prompt = "Answer the question in the image."
print(llama4(prompt, [f"data:image/png;base64,{base64_math}"]))

# %% [markdown]
# ## Analyzing computer screen

# %%
# Notebook cell 31
display_local_image("images/browser_screenshot.png")

# %%
# Notebook cell 32
base64_img = encode_image_to_base64("images/browser_screenshot.png")
prompt = """Describe the screenshot in detail,
including browser URL and tabs."""
print(llama4(prompt,[f"data:image/png;base64,{base64_img}"]))

# %%
# Notebook cell 33
display_local_image("images/browser_screenshot.png")
prompt = """If I want to go to the next lesson, what should I do?"""
print(llama4(prompt,[f"data:image/png;base64,{base64_img}"]))

# %%
# Notebook cell 34

