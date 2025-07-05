import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
from model import LoraCell
from torch.optim import AdamW
import numpy as np
import pickle
import wandb
from nltk import sent_tokenize
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments with text compression using LoRA adaptation')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-160m', help='Name of the model to use')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for computations')
    parser.add_argument('--use_flash_attention_2', action='store_true', help='Whether to use flash attention 2')
    parser.add_argument('--lora_r', type=int, nargs='+', default=[8, 16, 32, 64, 128],
                        help='List of LoRA rank values to experiment with')
    parser.add_argument('--lora_alpha', type=int, nargs='+', default=[1],#подобрать
                        help='List of LoRA alpha values to experiment with')
    parser.add_argument('--max_length', type=int, nargs='+',
                        default=[8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1568],
                        help='List of max lengths to experiment with')
    parser.add_argument('--num_iterations', type=int, default=5000, help='Number of iterations for each experiment')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use from the dataset')
    parser.add_argument('--lr', type=float, default=1e-02, help='learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='adam beta_1')
    parser.add_argument('--beta_2', type=float, default=0.9, help='adam beta_2')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=2000, help='Early stopping patience')
    parser.add_argument('--shuffled', action='store_true', help='Whether to use random text sampled from GloVe vocab.')
    parser.add_argument('--save_path', type=str, default='./runs', help='path to save experiments')
    parser.add_argument('--texts_path', type=str, default='./data/pg19_valid_1k_chunks.csv',
                        help='path to texts to compress')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate')#подобрать
    parser.add_argument('--target_modules', type=str, nargs='+', default=None,#подобрать
                        help='Target modules for LoRA adaptation')
    
    return parser.parse_args()


def calculate_accuracy(logits, labels):
    # bs = 1
    shift_logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    predictions = torch.argmax(shift_logits, dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()


def run_single_experiment(lora_r, lora_alpha, text_sample, max_length, num_iterations, sample_idx, run_idx,
                          model_name, dtype, use_flash_attention_2, device, tokenizer, lr, beta_1, beta_2,
                          weight_decay, early_stopping_patience=2000, shuffled=False, lora_dropout=0.1,
                          target_modules=None):
    # split text sample on two parts: prefix and main text
    sentences = sent_tokenize(text_sample)
    # prefix can be used lately for compression analysis
    # prefix_text = ' '.join(sentences[:len(sentences)//2])
    # suffix is compressed
    suffix_text = ' '.join(sentences[len(sentences)//2:])

    if shuffled:
        vocab = []
        with open('./data/vocab_100k.txt') as fin:
            for line in fin:
                vocab += [line.strip()]
        max_length = np.random.randint(2, max_length+1)
        suffix_text = ' '.join(np.random.choice(vocab, size=max_length * 5))
        inp = tokenizer(suffix_text, max_length=max_length, truncation=True, return_tensors='pt').to(device)
    else:
        inp = tokenizer(suffix_text, max_length=max_length, truncation=True, return_tensors='pt').to(device)

    model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=use_flash_attention_2)
    model.to(device)

    with torch.cuda.amp.autocast(dtype=dtype):
        with torch.no_grad():
            orig_output = model(**inp, labels=inp['input_ids'])
            orig_loss = orig_output.loss.item()
            orig_accuracy = calculate_accuracy(orig_output.logits, inp['input_ids'])

    model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=use_flash_attention_2)
    model_with_lora = LoraCell(model, lora_r=lora_r, lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, target_modules=target_modules)
    model_with_lora.to(device)

    opt = AdamW(model_with_lora.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2))

    desc = (f"Training (r={lora_r}, alpha={lora_alpha}, l={max_length}, i={sample_idx}), "
            f"no_lora_loss={orig_loss:.4f}, no_lora_acc={orig_accuracy:.4f}")
    progress_bar = tqdm(range(num_iterations), desc=desc, leave=False)


    losses, accuracies = [], []
    best_loss, best_accuracy, = float('inf'), 0
    best_lora_state = None
    early_stopping_counter = 0

    for _ in progress_bar:
        with torch.cuda.amp.autocast(dtype=dtype):
            out = model_with_lora(**inp)
            loss = out.loss
            accuracy = calculate_accuracy(out.logits, inp['input_ids'])

        loss.backward()
        opt.step()
        opt.zero_grad()
        current_loss = loss.item()
        losses.append(current_loss)
        accuracies.append(accuracy)

        if best_accuracy < accuracy:
            best_loss = current_loss
            best_accuracy = accuracy
            best_lora_state = {name: param.data.clone() for name, param in model_with_lora.named_parameters() if param.requires_grad}
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        progress_bar.set_postfix(loss=f"{current_loss:.4f}", best_loss=f"{best_loss:.4f}",
                                 best_acc=f"{best_accuracy:.4f}")

        if best_accuracy == 1.0:
            break

        if early_stopping_counter >= early_stopping_patience:
            break
        if wandb.run:
            wandb.run.log({
                'loss': current_loss,
                'accuracy': accuracy,
                'best_loss': best_loss,
                'best_accuracy': best_accuracy,
		'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'max_length': max_length,
                'sample_idx': sample_idx,
                'run_idx': run_idx,
                'iteration': _,
            })
    if wandb.run:
        wandb.run.log({
            'final_best_loss': best_loss,
            'final_best_accuracy': best_accuracy,
            'original_loss': orig_loss,
            'original_accuracy': orig_accuracy,
            'improvement_loss': orig_loss - best_loss,
            'improvement_accuracy': best_accuracy - orig_accuracy,
        })


    return {
	'losses': losses,
        'accuracies': accuracies,
        'original_loss': orig_loss,
        'original_accuracy': orig_accuracy,
        'best_lora_state': best_lora_state,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy,
        'max_length': max_length,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'suffix_text': suffix_text,
        'args': {
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'num_iterations': num_iterations,
            'sample_idx': sample_idx,
            'run_idx': run_idx,
            'model_name': model_name,
            'dtype': dtype,
            'use_flash_attention_2': use_flash_attention_2,
            'device': device,
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'weight_decay': weight_decay,
            'shuffled': shuffled,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules},
    }


def main():
    args = parse_arguments()
    wandb.init(
        project="hidden-capacity-lora",
        config={
            "model_name": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_length": args.max_length,
            "num_iterations": args.num_iterations,
            "num_samples": args.num_samples,
            "lr": args.lr,
            "beta_1": args.beta_1,
            "beta_2": args.beta_2,
            "weight_decay": args.weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "shuffled": args.shuffled,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        }
    )

    print(f'model: {args.model_name}')
    print(f'lora_r: {args.lora_r}')
    print(f'lora_alpha: {args.lora_alpha}')
    print(f'len: {args.max_length}')

    df = pd.read_csv(args.texts_path, index_col=0)

    device = 'cuda'
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    samples = df['text'][:args.num_samples]
    num_runs = 1

    total_experiments = len(args.max_length) * len(args.lora_r) * len(args.lora_alpha) * len(samples) * num_runs
    overall_progress = tqdm(total=total_experiments, desc="Overall Progress", position=0)

    for max_length in args.max_length:
        for lora_r in args.lora_r:
            for lora_alpha in args.lora_alpha:
                aggregated_results = []

                save_path = Path(f'./{args.save_path}/{args.model_name}')
                if not args.shuffled:
                    save_path = save_path / f'lora_r_{lora_r}_alpha_{lora_alpha}_len_{max_length}.pkl'
                else:
                    save_path = save_path / f'lora_r_{lora_r}_alpha_{lora_alpha}_len_{max_length}_rnd_vocab_100k.pkl'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                print(f'save_path: {save_path}')

                if save_path.exists():
                    print(f'loading previous results from {save_path}')
                    aggregated_results = pickle.load(open(save_path, 'rb'))

                for sample_idx, sample in enumerate(samples):
                    for run in range(num_runs):
                        result = run_single_experiment(lora_r, lora_alpha, sample, max_length, args.num_iterations, sample_idx,
                                                       run, args.model_name, dtype, args.use_flash_attention_2, device,
                                                       tokenizer, args.lr, args.beta_1, args.beta_2, args.weight_decay,
                                                       args.early_stopping_patience, args.shuffled, args.lora_dropout,
                                                       args.target_modules)
                        aggregated_results.append(result)
                        overall_progress.update(1)
                        pickle.dump(aggregated_results, open(save_path, 'wb'))


    overall_progress.close()


if __name__ == "__main__":
    main()
