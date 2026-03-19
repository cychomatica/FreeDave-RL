import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from termcolor import cprint
import os
from generation.fwd_counter import ForwardHookCounter
from generation.generate import DLM_Generator

def main(chat_history=False):
    
    # model_name = 'Dream-org/Dream-v0-Instruct-7B'
    model_name = 'Gen-Verse/TraDo-4B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(
    # model = AutoModel.from_pretrained(
        model_name, 
        torch_dtype='float16', 
        device_map='cuda',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    DLM = DLM_Generator(model)
    forward_counter = ForwardHookCounter(model)

    # Initialize conversation history
    messages = []

    print('Multi-turn conversation with {}'.format(os.path.basename(model_name)))
    print('Type ''exit'' to end the conversation')
    print('-'*100)
    
    while True:

        if not chat_history:
            messages = []

        prompt = input('Enter your question: \n')
        print('-'*100)

        # Check if user wants to exit
        if prompt.lower() == 'exit':
            print('Conversation ended.')
            break

        # Add user message to conversation history
        messages.append({'role': 'user', 'content': prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.batch_encode_plus(
            [text], return_tensors='pt', padding=True, truncation=True, max_length=200
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with forward_counter.count():
            output_ids = DLM.block_decode_with_block_causal_attention(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=4,
                max_gen_length=128,
                decoding_steps=128,
                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
            )
        output_ids = output_ids.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text = output_text.replace(tokenizer.special_tokens_map['mask_token'], '').replace(tokenizer.special_tokens_map['eos_token'], '').replace('<|im_end|>', '').strip()

        cprint(f'Normal generation: ({forward_counter})', 'yellow')
        print('Model\'s Response:', cleaned_text)
        print('-'*100)

        with forward_counter.count():
            output_ids, _ = DLM.block_decode_with_block_causal_attention_FreeDave(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=4,
                use_cache=True,
                max_gen_length=128,
                decoding_steps=128,
                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
                eager_acceptance_mode=True,
                draft_steps=8,
                draft_mode='batch_expanding',
            )
        output_ids = output_ids.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text = output_text.replace(tokenizer.special_tokens_map['mask_token'], '').replace(tokenizer.special_tokens_map['eos_token'], '').replace('<|im_end|>', '').strip()

        cprint(f'FreeDave generation (batch expanding): ({forward_counter})', 'green')
        print('Model\'s Response:', cleaned_text)
        print('-'*100)

        with forward_counter.count():
            output_ids, _ = DLM.block_decode_with_block_causal_attention_FreeDave(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=4,
                use_cache=True,
                max_gen_length=128,
                decoding_steps=128,
                mask_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['mask_token']],
                eos_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['eos_token']],
                pad_token_id=tokenizer.added_tokens_encoder[tokenizer.special_tokens_map['pad_token']],
                eager_acceptance_mode=True,
                draft_steps=8,
                draft_mode='tree_attention',
            )
        output_ids = output_ids.cpu()
        torch.cuda.empty_cache()

        output_text = tokenizer.decode(output_ids[0][len(tokens['input_ids'][0]):], skip_special_tokens=False)
        cleaned_text = output_text.replace(tokenizer.special_tokens_map['mask_token'], '').replace(tokenizer.special_tokens_map['eos_token'], '').replace('<|im_end|>', '').strip()

        cprint(f'FreeDave generation (tree attention): ({forward_counter})', 'green')
        print('Model\'s Response:', cleaned_text)
        print('-'*100)
        # Add the response from normal generation to the conversation history by default
        messages.append({'role': 'assistant', 'content': cleaned_text})

if __name__ == '__main__':
    main(chat_history=True)
