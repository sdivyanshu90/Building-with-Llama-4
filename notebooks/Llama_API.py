# %% [markdown]
# Source notebook: Llama_API.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L2: Llama API Quickstart

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Load API keys and libraries

# %% [markdown]
# **No need for a key on the platform:** The Llama 4 API key is already set up for you on the platform — you don’t need to provide one.
# 
# If you’d like to run the notebooks on your own and want to get your own key, follow these steps:
# 
# 1. Go to the [Llama API page](https://www.llama.com/products/llama-api/)
# 2. Click **“Join the waitlist.”**
# 3. After submitting your information, you'll receive an invite code via email.
# 4. Use that code to sign up — **you should get access right away**, based on our latest test.
# 
# > **Note:** At the time of this course recording, the API is in **preview mode** and may only be available to users in the **United States**. Availability may expand over time, so check the latest status on the Llama website.

# %%
# Notebook cell 6
import os
from utils import get_llama_api_key, get_llama_base_url

llama_api_key = get_llama_api_key()
llama_base_url = get_llama_base_url()

from llama_api_client import LlamaAPIClient

# %% [markdown]
# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# %% [markdown]
# ## Llama API client Python library

# %%
# Notebook cell 9
def llama4(prompt, image_urls=[],
    model="Llama-4-Scout-17B-16E-Instruct-FP8"  # or Llama-4-Maverick-17B-128E-Instruct-FP8
):
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append(
        {"type": "image_url", "image_url": {"url": url}})

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = LlamaAPIClient(api_key=llama_api_key)

  response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": content
    }],
    temperature=0
  )
  return response.completion_message.content.text

# %%
# Notebook cell 10
print(llama4("A brief history of AI in 3 short sentences."))

# %% [markdown]
# ## Using OpenAI-compatible library

# %%
# Notebook cell 12
from openai import OpenAI

def llama4(prompt,
    image_urls=[],
    model="Llama-4-Scout-17B-16E-Instruct-FP8",  # or Llama-4-Maverick-17B-128E-Instruct-FP8
    debug=False
):
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append(
        {"type": "image_url", "image_url": {"url": url}}) # TODO: for local image use {"url": "data:image/png;base64,..."}}

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = OpenAI(api_key=llama_api_key, base_url=llama_base_url)

  response = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "user",
        "content": content
    }],
    temperature=0
  )

  if debug:
    print(response)

  return response.choices[0].message.content

# %%
# Notebook cell 13
print(llama4("A brief history of AI in 3 short sentences."))

# %% [markdown]
# ## Asking question about a single image

# %%
# Notebook cell 15
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

# %%
# Notebook cell 16
img_url = "https://raw.githubusercontent.com/meta-llama/llama-models/refs/heads/main/Llama_Repo.jpeg"
display_image(img_url)

# %%
# Notebook cell 17
print(llama4("What's in the image?", [img_url]))

# %% [markdown]
# ## Asking question about multiple images

# %%
# Notebook cell 19
img_url2 = "https://raw.githubusercontent.com/meta-llama/PurpleLlama/refs/heads/main/logo.png"
display_image(img_url)
display_image(img_url2)

# %%
# Notebook cell 20
print(llama4("Compare these two images.", [img_url, img_url2]))

# %% [markdown]
# ## Llama 4 Long Context

# %%
# Notebook cell 22
with open("A-tale-of-two-cities.txt", "r", encoding='utf=8') as file:
    tale = file.read()

# %%
# Notebook cell 23
question = """What's the last utterance Sydney Carton would have given
at the end of the book A Tale of Two Cities? Just give one.
What's the paragraph before the last utterance?
"""

# %%
# Notebook cell 24
print(llama4(f"{question} The book content is below: {tale[300000:]}",
      model="Llama-4-Maverick-17B-128E-Instruct-FP8"))

# %% [markdown]
# ## Llama 4 Multilingual Support

# %%
# Notebook cell 26
question = """how many languages do you understand?
answer in all the languages you can speak."""

print(llama4(question))

# %%
# Notebook cell 27
client = OpenAI(api_key=llama_api_key, base_url=llama_base_url)

class Polyglot:
    def __init__(self, source="English", target="French", model="Llama-4-Scout-17B-16E-Instruct-FP8"):
        system = f"""You're a bilingual translator between two people:
          the first person only speaks {source} and
          the second person only speaks {target}.
          For any user input, return as follows:
          1. Recognized language: <the languege of the input,
          either {source} or {target}>
          2. Translation of the input: <the translation of the input to
          the other language
          3. Answer to the input: <in the same language as the
          recognized language of the input>"""
        self.messages = [{"role": "system", "content": system}]
        self.model = model

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        response = client.chat.completions.create(
                        model=self.model,
                        temperature=0,
                        messages=self.messages)
        return response.choices[0].message.content

# %%
# Notebook cell 28
agent = Polyglot(source="English", target="French")

# %%
# Notebook cell 29
result = agent("Hello!")
print(result)

# %%
# Notebook cell 30
print(agent("How do you say the weather is nice in French?"))

# %%
# Notebook cell 31

