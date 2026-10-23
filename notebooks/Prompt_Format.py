# %% [markdown]
# Source notebook: Prompt_Format.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L4: Prompt format

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load API Keys

# %%
# Notebook cell 5
import os
from utils import get_llama_api_key, get_hf_access_token

llama_api_key = get_llama_api_key()
hf_access_token = get_hf_access_token()

# %% [markdown]
# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# %% [markdown]
# ## Using Hugging Face transformers

# %%
# Notebook cell 8
from transformers import AutoProcessor

model_llama4_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
model_llama33_id = "meta-llama/Llama-3.3-70B-Instruct"

processor_llama4 = AutoProcessor.from_pretrained(model_llama4_id)
processor_llama33 = AutoProcessor.from_pretrained(model_llama33_id)

# %% [markdown]
# ## Comparing Llama 4 and 3 raw text prompts

# %%
# Notebook cell 10
messages=[{
    "role": "user",
    "content": "The best quote in Godfather."
}]

# %%
# Notebook cell 11
raw_prompt = processor_llama4.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True)
raw_prompt

# %%
# Notebook cell 12
raw_prompt = processor_llama33.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True)
raw_prompt

# %%
# Notebook cell 13
messages = [
    {"role": "system", "content": "Respond in French."},
    {"role": "user", "content": "Best quote in Godfather."},
]

# %%
# Notebook cell 14
raw_input_prompt = processor_llama4.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True)
raw_input_prompt

# %%
# Notebook cell 15
raw_input_prompt = processor_llama33.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True)
raw_input_prompt

# %% [markdown]
# ## Image prompt format

# %%
# Notebook cell 17
url = "https://raw.githubusercontent.com/meta-llama/llama-models/refs/heads/main/Llama_Repo.jpeg"

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def display_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

display_image(url)

# %%
# Notebook cell 18
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe the image below.",
            },
            {"type": "image", "url": url},
        ],
    },
]

# %%
# Notebook cell 19
inputs = processor_llama4.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# %%
# Notebook cell 20
inputs.keys()

# %%
# Notebook cell 21
inputs.pixel_values.size()

# %%
# Notebook cell 22
from utils import plot_tiled_image
plot_tiled_image(width=768, height=768, tile_size=336, patch_size=28)

# %%
# Notebook cell 23
raw_prompt = processor_llama4.tokenizer.batch_decode(inputs["input_ids"])
raw_prompt[0].replace("<|patch|>"*144, "<|patch|>...<|patch|>")

# %%
# Notebook cell 24

