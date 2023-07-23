# LLM_scripts

## Yes/No Questions with logit probability
This script allows entering prompt that are yes/no questions. The logit probability of the "yes" answer is returned.
The code should work for most tranformer language models on huggingface, but it was only tested here for Flan-T5.
It is currently configured for GPU machinces.
```
> python3 flan_t5_yesno_logit.py
Loading model...
Loading checkpoint shards: 100%|████████████████████████████████████| 5/5 [00:31<00:00,  6.38s/it]
Ask a yes or no question, and end it with a "?". Write "done" when finished.
Question: Is a dog a type of animal?
Logit probability "yes": 0.9892578125
Question: Is a dog a type of plant?
Logit probability "yes": 0.02734375
done
>
```
Notice that the logit is in the range of 0 and 1, but it is not linear. You should play around with the threshold to decide when to assign a yes or a no answer.
