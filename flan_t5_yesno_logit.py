import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np

class Model_flan_t5:
    def __init__(self, temperature=0.3):
        self.model_name = "google/flan-t5-xxl"
        self.set_seed()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float16, load_in_8bit=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.temperature = temperature
        
    def set_seed(self):
        # set seed
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def infer(self, prompt):
        # tokenize the prompt:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        # generate the output(s):
        outputs = self.model.generate(input_ids.to('cuda'), max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        # encoder-decoder models, like BART or T5.
        input_length = 1 if self.model.config.is_encoder_decoder else input_ids.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        prob_logits = np.exp(transition_scores[0][0].to('cpu').numpy())
        tok = self.tokenizer.decode(generated_tokens[0][0])
        if tok == 'no':
            prob_logit_yes = 1 - prob_logits
        elif tok == 'yes':
            prob_logit_yes = prob_logits
        else:
            prob_logit_yes = -1
        return prob_logit_yes
		
# ask a yes or no question and get the probability that it is "yes"
print('Loading model...')
model = Model_flan_t5()
print('Ask a yes or no question, and end it with a "?". Write "done" when finished.')
request = input('Question: ')
while request.lower() != 'done':
    prompt = request
    prompt += ' Answer yes or no.'
    prob_logit_yes = model.infer(prompt)
    if prob_logit_yes >= 0:
        print(f'Logit probability "yes": {prob_logit_yes}')
    else:
        print('Answer is not yes or no.')
    request = input('Question: ')