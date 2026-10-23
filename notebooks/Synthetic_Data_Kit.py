# %% [markdown]
# Source notebook: Synthetic_Data_Kit.ipynb
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
# # L7: Synthetic Data Kit

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
from utils import get_llama_api_key
llama_api_key = get_llama_api_key()

# %%
# Notebook cell 6
os.environ["API_ENDPOINT_KEY"] = llama_api_key

# %%
# Notebook cell 7
#!pip install synthetic-data-kit==0.0.4b1

# %% [markdown]
# <div style="background-color:#fff6ff; padding:13px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
# <p> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>.</p>
# 
# <p> ⬇ &nbsp; <b>Download Notebooks:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>.</p>
# 
# <p> 📒 &nbsp; For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
# </div>

# %% [markdown]
# # Ingeting PDF files and web pages

# %%
# Notebook cell 10
get_ipython().system('synthetic-data-kit ingest paper.pdf')

# %%
# Notebook cell 11
get_ipython().system('head -50 data/output/paper.txt | tail -10')

# %%
# Notebook cell 12
get_ipython().system('synthetic-data-kit ingest https://ai.meta.com/blog/llama-4-multimodal-intelligence/')

# %%
# Notebook cell 13
get_ipython().system('head -50 data/output/ai_meta_com.txt | tail -10')

# %% [markdown]
# ## Creating a QA dataset

# %%
# Notebook cell 15
get_ipython().system('synthetic-data-kit create data/output/paper.txt --type qa')

# %%
# Notebook cell 16
get_ipython().system('cat data/generated/paper_qa_pairs.json')

# %% [markdown]
# ## Curating the dataset

# %%
# Notebook cell 18
get_ipython().system('synthetic-data-kit curate data/generated/paper_qa_pairs.json --threshold=8')

# %%
# Notebook cell 19
get_ipython().system('cat data/cleaned/paper_qa_pairs_cleaned.json')

# %% [markdown]
# ## Saving the dataset

# %%
# Notebook cell 21
get_ipython().system('synthetic-data-kit save-as data/cleaned/paper_qa_pairs_cleaned.json --format jsonl --storage json')

# %%
# Notebook cell 22
get_ipython().system('head -10  data/final/paper_qa_pairs_cleaned.jsonl')

# %%
# Notebook cell 23
get_ipython().system('synthetic-data-kit save-as data/cleaned/paper_qa_pairs_cleaned.json --format ft --storage json')

# %%
# Notebook cell 24
get_ipython().system('head -30 data/final/paper_qa_pairs_cleaned_ft.json')

# %% [markdown]
# ## Configuration file

# %%
# Notebook cell 26
get_ipython().system('cat "$(pip show synthetic-data-kit | grep Location | awk \'{print $2}\')/synthetic_data_kit/config.yaml"')

# %% [markdown]
# ## Do it yourself: Creating a CoT dataset

# %% [markdown]
# In this section you can see how a Chain of Thought reasoning dataset can be created from the paper. By default, 10 CoT examples will be created. You may change it either by using a command line parameter `--num-pairs` as shown below.

# %%
# Notebook cell 29
get_ipython().system('synthetic-data-kit create data/output/paper.txt --type cot --num-pairs 5')

# %%
# Notebook cell 30
get_ipython().system('cat  data/generated/paper_cot_examples.json')

# %% [markdown]
# Each created `cot_examples` is a dictionary with 3 keys: `question`, `reasoning` and `answer`. For example:
# 
# ```
#       "question": "How does the Byte Latent Transformer (BLT) architecture dynamically allocate compute based on data complexity?",
# 
#       "reasoning": "Step 1: Understand that BLT uses a dynamic method for grouping bytes into patches.\nStep 2: Recognize that the patching function segments bytes based on the entropy of the next byte prediction.\nStep 3: Analyze how the entropy patching method uses a small byte-level language model to compute next byte entropies.\nStep 4: Determine how patch boundaries are identified based on entropy thresholds or relative changes in entropy.\nStep 5: Conclude that BLT dynamically allocates compute by invoking the Latent Transformer based on patch boundaries determined by entropy.",
#       
#       "answer": "BLT dynamically allocates compute by segmenting bytes into patches based on the entropy of the next byte prediction, using a small byte-level language model to determine patch boundaries."
# ```
# 
# If the `reasoning` steps above are not obvious from `question` to `answer`, below is a grade level math reasoning example for you to easily verify the correctness of reasoning.

# %% [markdown]
# ## Do it yourself: Creating a math reasoning dataset
# 
# GSM8K is a dataset of 8500 high quality linguistically diverse grade school math word problems. To use it you need to run: `pip install -U datasets==2.14.6`. Run the code below to get 50 examples from the dataset and save the questions in the examples to a text file.

# %%
# Notebook cell 33
import pandas as pd
import os
from datasets import load_dataset

# Create directories if they don't exist
os.makedirs('data/output', exist_ok=True)

# Load GSM8K dataset
gsm8k = load_dataset('gsm8k', 'main')

# Take 50 samples from the training set
samples = gsm8k['train'].select(range(50))

# Create a text file with the questions
with open('data/output/gsm8k_sample.txt', 'w') as f:
    for i, item in enumerate(samples):
        f.write(f"Problem {i+1}:\n{item['question']}\n\n")

print(f"Created sample with {len(samples)} problems")
print(f"Sample saved to data/output/gsm8k_sample.txt")

# %% [markdown]
# The first 3 examples of the dataset's questions are:
# 
# ```
# Problem 1:
# Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
# 
# Problem 2:
# Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
# 
# Problem 3:
# Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
# ```

# %%
# Notebook cell 35
get_ipython().system('cat data/output/gsm8k_sample.txt')

# %% [markdown]
# Now run the data kit tool to create a math reasoning <code>cot</code> dataset from the text file.

# %%
# Notebook cell 37
get_ipython().system('synthetic-data-kit create data/output/gsm8k_sample.txt --type cot')

# %%
# Notebook cell 38
get_ipython().system('cat data/generated/gsm8k_sample_cot_examples.json')

# %% [markdown]
# You can check two examples of the generated dataset and confirm the added reasoning steps are correct:

# %%
# Notebook cell 40
import glob
json_files = glob.glob("data/generated/gsm8k_sample_cot_examples.json")

import json
with open(json_files[0], "r") as f:
    data = json.load(f)
print(data['cot_examples'][0]['question'])
print(data['cot_examples'][0]['reasoning'])
print(data['cot_examples'][0]['answer'])

# %%
# Notebook cell 41
print(data['cot_examples'][-1]['question'])
print(data['cot_examples'][-1]['reasoning'])
print(data['cot_examples'][-1]['answer'])

# %%
# Notebook cell 42

