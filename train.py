import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
from model import MemoryCell
from torch.optim import AdamW
import numpy as np
import pickle
from nltk import sent_tokenize
from pathlib import Path
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments with text compression using memory tokens')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-160m', help='Name of the model to use')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'],
                        help='Data type for computations')
    parser.add_argument('--use_flash_attention_2', action='store_true', help='Whether to use flash attention 2')
    parser.add_argument('--N_mem_tokens', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help='List of memory token numbers to experiment with')
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
    return parser.parse_args()


def calculate_accuracy(logits, labels):
    # bs = 1
    shift_logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    predictions = torch.argmax(shift_logits, dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()

def setup_logging(save_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_path / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_single_experiment(N_mem_tokens, text_sample, max_length, num_iterations, sample_idx, run_idx,
                              model_name, dtype, use_flash_attention_2, device, tokenizer, lr, beta_1, beta_2,
                              weight_decay, early_stopping_patience=2000, shuffled=False, logger=None):

    # Init wandb
    wandb.init(
        project="no-density-experiment",
        name=f"run_{run_idx}_sample_{sample_idx}_mem_{N_mem_tokens}",
        config={
            "N_mem_tokens": N_mem_tokens,
            "max_length": max_length,
            "num_iterations": num_iterations,
            "sample_idx": sample_idx,
            "run_idx": run_idx,
            "model_name": model_name,
            "dtype": str(dtype),
            "use_flash_attention_2": use_flash_attention_2,
            "device": str(device),
            "lr": lr,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "weight_decay": weight_decay,
            "shuffled": shuffled,
        }
    )

    if logger:
        logger.info(f"Starting experiment with N_mem_tokens={N_mem_tokens}, max_length={max_length}, "
                    f"sample_idx={sample_idx}, run_idx={run_idx}")

    sentences = sent_tokenize(text_sample)
    suffix_text = ' '.join(sentences[len(sentences) // 2:])

    if shuffled:
        vocab = []
        with open('./data/vocab_100k.txt') as fin:
            for line in fin:
                vocab += [line.strip()]
        max_length = np.random.randint(2, max_length + 1)
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

    if logger:
        logger.info(f"Original loss: {orig_loss:.4f}, Original accuracy: {orig_accuracy:.4f}")

    wandb.log({"original_loss": orig_loss, "original_accuracy": orig_accuracy})

    model = AutoModelForCausalLM.from_pretrained(model_name, use_flash_attention_2=use_flash_attention_2)
    memory_dim = getattr(model.config, 'word_embed_proj_dim', getattr(model.config, 'hidden_size'))
    model_with_memory = MemoryCell(model, N_mem_tokens, memory_dim)
    model_with_memory.to(device)

    opt = AdamW(model_with_memory.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2))

    desc = (f"Training (m={N_mem_tokens}, l={max_length}, i={sample_idx}), "
            f"no_mem_loss={orig_loss:.4f}, no_mem_acc={orig_accuracy:.4f}")
    progress_bar = tqdm(range(num_iterations), desc=desc, leave=False)

    losses, accuracies = [], []
    best_loss, best_accuracy, = float('inf'), 0
    best_memory_params = None
    early_stopping_counter = 0
    log_interval = max(1, num_iterations // 100)  # Логируем примерно 100 раз за эксперимент
    reconstruction_log_interval = max(1, num_iterations // 25)

    for step in progress_bar:
        with torch.cuda.amp.autocast(dtype=dtype):
            out, mem = model_with_memory(**inp)
            loss = out.loss
            accuracy = calculate_accuracy(out.logits, inp['input_ids'])

        loss.backward()
        opt.step()
        opt.zero_grad()
        current_loss = loss.item()
        losses.append(current_loss)
        accuracies.append(accuracy)

        # Логирование на каждой итерации в WandB
        wandb.log({
            "step": step,
            "loss": current_loss,
            "accuracy": accuracy,
            "best_accuracy": best_accuracy,
            "best_loss": best_loss
        })

        # Логирование в консоль и файл через регулярные интервалы
        if step % log_interval == 0 and logger:
            logger.info(f"Iteration {step}/{num_iterations}: "
                        f"Loss={current_loss:.4f}, Accuracy={accuracy:.4f}, "
                        f"Best Loss={best_loss:.4f}, Best Accuracy={best_accuracy:.4f}")
        if step % reconstruction_log_interval == 0 and logger:
            # Получаем реконструированный текст
            preds = torch.argmax(out.logits, dim=-1)
            reconstructed_text = tokenizer.decode(preds[0], skip_special_tokens=True)
            logger.info(f"Reconstructed text at iteration {step}: {reconstructed_text[:500]}")  # первые 500 символов
            wandb.log({f"reconstructed_text_step_{step}": reconstructed_text[:500]})

        if best_accuracy < accuracy:
            best_loss = current_loss
            best_accuracy = accuracy
            best_memory_params = model_with_memory.memory.data.cpu().numpy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        progress_bar.set_postfix(loss=f"{current_loss:.4f}", best_loss=f"{best_loss:.4f}",
                                 best_acc=f"{best_accuracy:.4f}")

        if best_accuracy == 1.0:
            if logger:
                logger.info("Early stopping as perfect accuracy reached")
            break

        if early_stopping_counter >= early_stopping_patience:
            if logger:
                logger.info(f"Early stopping after {early_stopping_patience} iterations without improvement")
            break

    preds = torch.argmax(out.logits, dim=-1)
    final_reconstructed_text = tokenizer.decode(preds[0], skip_special_tokens=True)
    logger.info(f"Final reconstructed text: {final_reconstructed_text[:1000]}")
    wandb.log({"final_reconstructed_text": final_reconstructed_text[:1000]})

    wandb.log({
        "final_best_accuracy": best_accuracy,
        "final_best_loss": best_loss
    })
    wandb.finish()

    if logger:
        logger.info(f"Experiment completed. Best loss: {best_loss:.4f}, Best accuracy: {best_accuracy:.4f}")

    return {
        'losses': losses,
        'accuracies': accuracies,
        'original_loss': orig_loss,
        'original_accuracy': orig_accuracy,
        'best_memory_params': best_memory_params,
        'best_loss': best_loss,
        'best_accuracy': best_accuracy,
        'max_length': max_length,
        'n_mem_tokens': N_mem_tokens,
        'suffix_text': suffix_text,
        'args': {
            'N_mem_tokens': N_mem_tokens,
            'max_length': max_length,
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
            'shuffled': shuffled},
    }


def main():
    args = parse_arguments()

    # Создаем уникальную папку для этого запуска
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    if args.shuffled:
        run_folder += "_shuffled"

    # Базовый путь для сохранения
    base_save_path = Path(args.save_path) / args.model_name / run_folder
    base_save_path.mkdir(parents=True, exist_ok=True)

    # Настройка логирования
    logger = setup_logging(base_save_path)

    logger.info(f"Starting experiment with parameters:\n{args}")
    logger.info(f"Results will be saved to: {base_save_path}")

    df = pd.read_csv(args.texts_path, index_col=0)
    device = 'cuda'
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    samples = df['text'][:args.num_samples]
    num_runs = 1

    total_experiments = len(args.max_length) * len(args.N_mem_tokens) * len(samples) * num_runs
    overall_progress = tqdm(total=total_experiments, desc="Overall Progress", position=0)

    # Сохраняем промежуточные результаты каждые N экспериментов
    save_interval = max(1, len(samples) // 10)  # Сохраняем примерно 10 раз за полный проход

    for max_length in args.max_length:
        for N_mem_tokens in args.N_mem_tokens:
            if N_mem_tokens > max_length:
                continue

            aggregated_results = []
            save_path = base_save_path / f'mem_{N_mem_tokens}_len_{max_length}.pkl'

            if save_path.exists():
                logger.info(f'Loading previous results from {save_path}')
                aggregated_results = pickle.load(open(save_path, 'rb'))

            for sample_idx, sample in enumerate(samples):
                for run in range(num_runs):
                    result = run_single_experiment(
                        N_mem_tokens, sample, max_length, args.num_iterations, sample_idx,
                        run, args.model_name, dtype, args.use_flash_attention_2, device,
                        tokenizer, args.lr, args.beta_1, args.beta_2, args.weight_decay,
                        args.early_stopping_patience, args.shuffled, logger
                    )
                    aggregated_results.append(result)
                    overall_progress.update(1)

                    # Периодическое сохранение промежуточных результатов
                    if sample_idx % save_interval == 0:
                        pickle.dump(aggregated_results, open(save_path, 'wb'))
                        logger.info(f"Intermediate results saved to {save_path}")

            # Финальное сохранение после завершения всех samples
            pickle.dump(aggregated_results, open(save_path, 'wb'))
            logger.info(f"Final results for mem={N_mem_tokens}, len={max_length} saved to {save_path}")

    overall_progress.close()

    # Сохраняем аргументы запуска
    args_dict = vars(args)
    args_save_path = base_save_path / 'run_args.pkl'
    pickle.dump(args_dict, open(args_save_path, 'wb'))
    logger.info(f'Run arguments saved to: {args_save_path}')
    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    main()