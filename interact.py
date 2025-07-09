import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str, default=os.path.join("auto_coding", "model", "gpt2_medium_fine_tuned_coder"),
                        help='the path to load fine-tuned model')
    parser.add_argument('--max_length', type=int, default=128,
                        help='maximum length for code generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='temperature for sampling-based code generation')
    parser.add_argument('--use_cuda', action="store_true", help="inference with gpu?")
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path, local_files_only=True)
    model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    if args.use_cuda:
        model.to("cuda")

    def lang_select():
        lang = ""
        while lang not in ["python", "java"]:
            print('Enter the programming language you prefer (python or java)')
            lang = input(">>> ").lower()
        return lang

    lang = lang_select()
    context = ""
    while context != "exit":
        print(f'You are using {lang} now. Enter the context code (exit or change_lang)')
        context = input(">>> ")
        if context == "change_lang":
            lang = lang_select()
            print(f"You are using {lang} now. Enter the context code")
            context = input(">>> ")

        prompt = f"<{lang}> {context}"
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = input_ids.ne(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None

        outputs = model.generate(
            input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
            attention_mask=attention_mask.to("cuda") if args.use_cuda else attention_mask,
            max_length=args.max_length,
            temperature=args.temperature,
            num_return_sequences=1
        )

        for i in range(1):
            decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if "\n\n" in decoded:
                decoded = decoded[:decoded.index("\n\n")]
            print('Generated {}: {}'.format(i, decoded))
