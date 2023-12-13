# Supplementary Experiments.
We apply COSTELLO to test [BART](https://huggingface.co/facebook/bart-large-mnli),[T5](https://huggingface.co/google/flan-t5-base), [LLaMA2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [OpenAI Embedding API](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings).
## How to run
We have collected the embeddings from those models or APIs. You can `cd` into the dir of {BART, LLaMA2, T5, OpenAI} and run 'bash script.sh' to perform the testing and evaluation.
```
cd dir  
bash script.sh
```
## Results
Here is our evaluation result:

![Results](./results.png)

COSTELLO also exhibits qualified effectiveness onth BART and T5. There's a decline in test precision on embeddings provided by LLaMA2 and particularly OpenAI.
However, this observation suggests that OpenAI potentially provides higher-quality embeddings, resulting in downstream applications achieve superior performance with fewer violations of contrastive relationships. 
This aligns well with our expectations regarding the current state-of-the-art LLMs, as they have demonstrated remarkable performance across various tasks.

The decline in test precision might be attributed to higher-dimensional embeddings provided by OpenAI and LLaMA2, which pose more challenges. Additionally, these latest LLMs utilize more complex tokenizers, potentially rendering our lexicon (which is used for threshold computation) less suitable for them.
In our future plan, we aim to enhance testing effectiveness by exploring more adaptable lexicons tailored for recent LLMs.