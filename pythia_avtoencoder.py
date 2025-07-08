import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import datasets
from nltk import sent_tokenize
from tqdm import tqdm
import random
import wandb

def get_random_chunk(text, min_chunk_size):
    text = ' '.join(text.split())
    sentences = sent_tokenize(text)
    if len(sentences) < min_chunk_size:
        return ' '.join(sentences)  # Return the entire text if it's shorter than min_chunk_size
    n_words = 0
    while not (n_words < 25000 and n_words > 12000):
        max_start = len(sentences) - min_chunk_size
        start = random.randint(0, max_start)
        end = random.randint(start + min_chunk_size, len(sentences))
        chunk = ' '.join(sentences[start:end])
        n_words = len(chunk.split())
    return chunk

class PythiaAE(torch.nn.Module):
    """Autoencoder with two copies of Pythia-160M (Encoder & Decoder)."""

    def __init__(self, model_name: str = "EleutherAI/pythia-160m", latent_size: int = 768):
        super().__init__()

        # Load encoder/decoder
        self.encoder = AutoModelForCausalLM.from_pretrained(model_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze decoder weights
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.latent_size = latent_size

        # AE projections: hidden -> latent -> hidden
        self.to_latent = torch.nn.Linear(self.hidden_size, latent_size)
        self.latent_to_hidden = torch.nn.Linear(latent_size, self.hidden_size)

    @staticmethod
    def _mask_first(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """Prepends `ignore_index` to labels so the latent token is ignored in loss."""
        bsz = labels.size(0)
        pad = torch.full((bsz, 1), ignore_index, dtype=labels.dtype, device=labels.device)
        return torch.cat([pad, labels], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        # ---- Encoder ----
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = enc_out.hidden_states[-1]
        pooled = last_hidden[:, -1, :]  # CLS-like pooling using final token

        # No KL, just deterministic projection
        latent = self.to_latent(pooled)
        latent_emb = self.latent_to_hidden(latent).unsqueeze(1)  # (B, 1, H)

        # ---- Prepare decoder inputs ----
        embed_layer = self.decoder.get_input_embeddings()  # shared word-emb table
        token_embeds = embed_layer(input_ids)  # (B, L, H)
        inputs_embeds = torch.cat([latent_emb, token_embeds], dim=1)  # (B, L+1, H)

        # Extend attention mask (latent token is visible)
        dec_attention_mask = torch.cat(
            [torch.ones_like(attention_mask[:, :1]), attention_mask], dim=1
        )

        # Prepare labels: ignore latent position
        if labels is None:
            labels = input_ids  # teacher forcing defaults to original ids
        dec_labels = self._mask_first(labels)

        # ---- Decoder ----
        dec_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            labels=dec_labels,
            return_dict=True,
        )
        recon_loss = dec_out.loss
        logits = dec_out.logits

        # Total loss == только reconstruction
        total_loss = recon_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss.detach(),
            "logits": logits,
        }


# -------------------------------------------------------------------------
# Minimal training stub
# -------------------------------------------------------------------------

def train_one_epoch(model: PythiaAE, dataloader, optimizer, device: str = "cuda"):
    model.train().to(device)
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        outputs["loss"].backward()
        optimizer.step()
        total_loss += outputs["loss"].item()

        # Accuracy computation (with skipping first token in logits)
        logits = outputs["logits"][:, 1:, :]
        preds = torch.argmax(logits, dim=-1)
        mask = (input_ids != model.tokenizer.pad_token_id)
        correct_predictions += torch.sum((preds == input_ids) & mask).item()
        total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    wandb.log({"epoch_loss": avg_loss, "epoch_accuracy": accuracy})
    print(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

def evaluate_one_epoch(model, dataloader, device="cuda"):
    model.eval().to(device)
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += outputs["loss"].item()

            logits = outputs["logits"][:, 1:, :]
            preds = torch.argmax(logits, dim=-1)
            mask = (input_ids != model.tokenizer.pad_token_id)
            correct_predictions += torch.sum((preds == input_ids) & mask).item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions
    wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy})
    print(f"Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")
    return avg_loss, accuracy

def save_model(model, path="vae_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Загрузка модели
def load_model(model, path="vae_model.pt", device="cuda"):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"Model loaded from {path}")
    return model

def prepare_dataset(
    dataset_name: str = "deepmind/pg19",
    cache_dir: str = "data_cache",
    chunk_size: int = 256,      
    n_chunks: int = 1000,
):
    from nltk import sent_tokenize
    cache_file = Path(cache_dir) / f"pg19_token_chunks_{chunk_size}.pkl"
    if cache_file.exists():
        print("Loading cached token chunks...")
        return pickle.load(open(cache_file, "rb"))

    print("Processing token chunks...")
    ds = datasets.load_dataset(dataset_name, split="validation")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

    chunks = []
    for _ in tqdm(range(n_chunks), desc="Sampling token chunks"):
        text = random.choice(ds)["text"]
        sentences = sent_tokenize(text)
        random.shuffle(sentences)
        chunk = ''
        for sent in sentences:
            if len(tokenizer.encode(chunk + sent, add_special_tokens=False)) < chunk_size:
                chunk += ' ' + sent
            else:
                break
        if len(chunk.strip()) < 50:
            continue
        chunks.append(chunk.strip())

    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(chunks, f)
    return chunks


if __name__ == "__main__":
    # Initialize W&B
    wandb.init(project="AutoEncoder_training_experiment")

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4

    # 1) Load and preprocess data
    texts = prepare_dataset()

    # 2) Split into train / val
    random.shuffle(texts)
    split = int(0.8 * len(texts))
    train_texts = texts[:split]
    val_texts   = texts[split:]

    # 3) Initialize model and rescale embeddings if needed
    ae = PythiaAE().to(device)
    if ae.tokenizer.pad_token is None:
        ae.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        ae.encoder.resize_token_embeddings(len(ae.tokenizer))
        ae.decoder.resize_token_embeddings(len(ae.tokenizer))

    # 4) Tokenize train and val separately
    train_enc = ae.tokenizer(
        train_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )
    val_enc = ae.tokenizer(
        val_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )

    # 5) Create TensorDatasets and DataLoaders
    train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"])
    val_ds   = TensorDataset(  val_enc["input_ids"],   val_enc["attention_mask"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 6) Optimizer and training loop setup
    optimizer = torch.optim.AdamW(ae.parameters(), lr=5e-5)
    best_loss = float("inf")
    num_epochs = 3000

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}")

        # — Training on train_loader
        train_one_epoch(ae, train_loader, optimizer, device)

        # — Validation on val_loader
        val_loss, val_acc = evaluate_one_epoch(ae, val_loader, device)

        # — Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(ae, path="best_ae_model.pt")

        # — Log a few reconstructions from the validation set
        ae.eval()
        table = wandb.Table(columns=["original", "reconstruction"])
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(val_loader, desc="Logging reconstructions"):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                out = ae(input_ids=input_ids, attention_mask=attention_mask)
                logits = out["logits"][:, 1:, :]
                preds = torch.argmax(logits, dim=-1)

                originals = [ae.tokenizer.decode(ids.cpu(), skip_special_tokens=True)
                             for ids in input_ids]
                recons    = [ae.tokenizer.decode(p.cpu(),   skip_special_tokens=True)
                             for p   in preds]

                for o, r in zip(originals, recons):
                    table.add_data(o, r)

        wandb.log({
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            f"reconstructions_epoch_{epoch+1}": table
        })

        ae.train()

