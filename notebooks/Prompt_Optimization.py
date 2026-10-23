# %% [markdown]
# Source notebook: Prompt_Optimization.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %% [markdown]
# # L6: Prompt Optimization

# %% [markdown]
# ## Load API Keys

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 4
import warnings
warnings.filterwarnings('ignore')

# %%
# Notebook cell 5
import os
from utils import get_llama_api_key, get_llama_base_url, get_together_api_key

llama_api_key = get_llama_api_key()
llama_base_url = get_llama_base_url()
together_api_key = get_together_api_key()

# %%
# Notebook cell 6
#!pip install llama-prompt-ops==0.0.7

# %% [markdown]
# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# %% [markdown]
# ## Creating a sample project

# %%
# Notebook cell 9
## Check if the folder exists: In the line [ -d "my-project" ] returns true if the directory is present; the || (“or”) means llama-prompt-ops create my-project executes only when that first test is false
get_ipython().system('[ -d "my-project" ] || llama-prompt-ops create my-project')

# %%
# Notebook cell 10
get_ipython().system('ls ./my-project/')

# %% [markdown]
# ## System prompt and dataset

# %%
# Notebook cell 12
get_ipython().system('cat my-project/prompts/prompt.txt')

# %%
# Notebook cell 13
get_ipython().system('head -15 my-project/data/dataset.json')

# %%
# Notebook cell 14
get_ipython().system('cat my-project/config.yaml')

# %%
# Notebook cell 15
get_ipython().run_cell_magic('writefile', 'my-project/config.yaml', 'system_prompt:\n  file: prompts/prompt.txt\n  inputs:\n  - question\n  outputs:\n  - answer\ndataset:\n  path: data/dataset.json\n  input_field:\n  - fields\n  - input\n  golden_output_field: answer\nmodel:\n  task_model: together_ai/meta-llama/Llama-4-Scout-17B-16E-Instruct\n  proposer_model: together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo\n  api_base: https://api.together.xyz/v1\nmetric:\n  class: llama_prompt_ops.core.metrics.FacilityMetric\n  strict_json: false\n  output_field: answer\noptimization:\n  strategy: llama\n')

# %% [markdown]
# ## Running prompt optimization

# %% [markdown]
# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p>Running prompt optimization can take a long time. To speed up running the notebooks, we will load saved results. You can change <code>run_optimization</code> to <code>True</code> to run the optimization.</p>
# </div>

# %%
# Notebook cell 18
run_optimization = False     # or chnage to True to run

if run_optimization:
    get_ipython().system('cd my-project && llama-prompt-ops migrate --api-key-env TOGETHERAI_API_KEY')

# %% [markdown]
# ## Analyzing the results

# %%
# Notebook cell 20
import glob
json_files = glob.glob("my-project/results/*.json")

import json
with open(json_files[0], "r") as f:
    data = json.load(f)
optimized_prompt = data['prompt']

# %%
# Notebook cell 21
with open("my-project/prompts/prompt.txt", "r", encoding="utf-8") as file:
    original_prompt = file.read()

# %%
# Notebook cell 22
from IPython.display import display, HTML

def compare_strings_side_by_side(text1, text2):
    html = '<table style="width: 100%; border-collapse: collapse;"><tr><th>Original Prompt</th><th>Optimized Prompt</th></tr>'
    html += f'<tr><td style="width:50% padding: 10px; vertical-align: top;"><pre style="white-space: pre-wrap; word-wrap: break-word;">{text1}</pre></td><td style="width: 50% padding: 10px; vertical-align: top;"><pre style="white-space: pre-wrap; word-wrap: break-word;">{text2}</pre></td></tr></table>'

    display(HTML(html))

compare_strings_side_by_side(original_prompt, optimized_prompt)

# %% [markdown]
# ## Few-shot examples

# %%
# Notebook cell 24
data['few_shots'][0]['question']

# %%
# Notebook cell 25
data['few_shots'][0]['answer']

# %%
# Notebook cell 26
len(data['few_shots'])

# %%
# Notebook cell 27
few_shots = "\n\nFew shot examples\n\n"
for i, shot in enumerate(data['few_shots']):
  few_shots += f"""Example {i+1}\n=================\nQuestion:\n
  {shot['question']}\n\nAnswer:\n{shot['answer']}\n\n"""

# %% [markdown]
# ## Compare optimized and original prompt

# %%
# Notebook cell 29
with open("my-project/data/dataset.json", 'r') as f:
  ds = json.load(f)

len(ds)

# %%
# Notebook cell 30
ds_test = ds[int(len(ds)*0.7):]
len(ds_test)

# %%
# Notebook cell 31
from utils import evaluate

# %%
# Notebook cell 32
from together import Together
from tqdm.auto import tqdm

result_original = []
client = Together()

for entry in tqdm(ds_test):
    messages=[
        {"role": "system", "content": original_prompt},
        {"role": "user", "content": entry["fields"]["input"]},
    ]

    response = client.chat.completions.create(
      model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
      messages=messages,
      temperature=0
    )

    prediction = response.choices[0].message.content
    result_original.append(evaluate(entry["answer"], prediction))

# %%
# Notebook cell 33
result_optimized = []

for entry in tqdm(ds_test):
    messages=[
        {"role": "system", "content": optimized_prompt + few_shots},
        {"role": "user", "content": entry["fields"]["input"]},
    ]

    response = client.chat.completions.create(
      model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
      messages=messages,
      temperature=0
    )

    prediction = response.choices[0].message.content
    result_optimized.append(evaluate(entry["answer"], prediction))

# %%
# Notebook cell 34
result_optimized[0]

# %%
# Notebook cell 35
result_optimized[1]

# %%
# Notebook cell 36
float_keys = [k for k, v in result_original[0].items() if isinstance(v,
                                                (int, float, bool))]
{k: sum([e[k] for e in result_original])/len(result_original) for k in float_keys}

# %%
# Notebook cell 37
{k: sum([e[k] for e in result_optimized])/len(result_optimized) for k in float_keys}

# %%
# Notebook cell 38

