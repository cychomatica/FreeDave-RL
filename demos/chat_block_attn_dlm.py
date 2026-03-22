import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from termcolor import cprint
import os
from generation.fwd_counter import ForwardHookCounter
from generation.generate import DLMGeneration
import argparse

def main(args):
    
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype='float16', 
        device_map='cuda',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    DLM_Gen = DLMGeneration()
    forward_counter = ForwardHookCounter(model)

    # Initialize conversation history
    messages = []

    print('Multi-turn conversation with {}'.format(os.path.basename(model_name)))
    print('Type ''exit'' to end the conversation')
    print('-'*100)
    
    while True:

        if not args.chat_history:
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
            output_ids, trajectory = DLM_Gen.block_decode_with_block_attention(
                model=model,
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=4,
                max_gen_length=256,
                decoding_steps=256,
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
            output_ids, _ = DLM_Gen.block_decode_with_block_attention_FreeDave(
                model=model,
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=4,
                use_cache=True,
                max_gen_length=256,
                decoding_steps=256,
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
            output_ids, _ = DLM_Gen.block_decode_with_block_attention_FreeDave(
                model=model,
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask'],
                temperature=0.0,
                top_p=None,
                top_k=None,
                alg_temp=None,
                block_length=4,
                use_cache=True,
                max_gen_length=256,
                decoding_steps=256,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--chat_history', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default='Gen-Verse/TraDo-4B-Instruct')
    args = parser.parse_args()
    main(args)
