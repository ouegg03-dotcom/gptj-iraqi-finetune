"""Fine-tune GPT-J (EleutherAI/gpt-j-6B) on the bot's Iraqi teachings with QLoRA.

SATURN VERSION — same as v2, but checkpoints go to a LOCAL folder
./gptj-iraqi-checkpoints (Saturn has no Google Drive). The script RESUMES
from the latest local checkpoint if it is restarted — a disconnect no
longer loses all progress, as long as the checkpoint folder is kept
(save a notebook version / attach the previous output).
  2. FASTER: per-device batch 8 (was 4), gradient checkpointing OFF
     (was ON), group_by_length ON. Same data, same 2 epochs, same
     effective batch 16. Roughly 25-40% faster than v1 (~7h instead of ~11h
     for 1204 steps). If it ever runs out of GPU memory, set
     BATCH=4 and CHECKPOINTING=True below.

Run on Saturn with a GPU accelerator (T4) + Internet ON:
  1. Upload training_data.jsonl + this file to the notebook.
  2. !pip install -q transformers==4.47.1 peft==0.13.2 trl==0.11.4 bitsandbytes==0.50.2 accelerate==0.34.2 datasets==3.0.1
  3. !python finetune_gptj_saturn.py
     To resume after a disconnect, just run the same command again — it
     picks up the latest checkpoint from ./gptj-iraqi-checkpoints automatically
     (keep that folder: save a notebook version before the session ends).

What it does:
  - loads GPT-J in 4-bit (NF4) so it fits on a 16GB T4
  - trains a small LoRA adapter (r=16) on training_data.jsonl (9640 examples)
  - saves checkpoints every 150 steps, each copied to ./gptj-iraqi-checkpoints
  - saves the adapter to ./gptj-iraqi-lora (in the notebook working dir)
  - tests the adapter on 3 Iraqi questions
The adapter is YOUR model: GPT-J + your teachings. To use it, load the base
model and attach the adapter (see the test code at the bottom).
"""
import glob
import os
import shutil

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

MODEL = "EleutherAI/gpt-j-6B"
DATA = "training_data.jsonl"
OUT = "./gptj-iraqi-lora"
EPOCHS = 2
BATCH = 8                 # micro-batch; drop to 4 if you hit out-of-memory
ACCUM = 2                 # effective batch = BATCH * ACCUM = 16 (same as v1)
CHECKPOINTING = False     # True = slower but less GPU memory
SAVE_STEPS = 150          # checkpoint every 150 optimizer steps
DRIVE_CKPT = "./gptj-iraqi-checkpoints"  # local folder on Saturn (no Drive)


def _step_num(path):
    try:
        return int(path.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


class DriveCheckpointCallback(TrainerCallback):
    """Copy each new checkpoint to Drive so a disconnect cannot lose it."""

    def on_save(self, args, state, control, **kwargs):
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(DRIVE_CKPT, f"checkpoint-{state.global_step}")
        if os.path.isdir(src) and not os.path.isdir(dst):
            shutil.copytree(src, dst)
            print("Checkpoint copied to:", dst)
            # keep only the latest 2 checkpoints here
            all_ck = sorted(
                glob.glob(os.path.join(DRIVE_CKPT, "checkpoint-*")), key=_step_num
            )
            for old in all_ck[:-2]:
                shutil.rmtree(old, ignore_errors=True)


def find_resume_checkpoint():
    """Return a local checkpoint dir to resume from, restoring it from Drive."""
    if not os.path.isdir(DRIVE_CKPT):
        return None
    ckpts = sorted(glob.glob(os.path.join(DRIVE_CKPT, "checkpoint-*")), key=_step_num)
    if not ckpts:
        return None
    latest = ckpts[-1]
    local = os.path.join("./gptj-iraqi-out", os.path.basename(latest))
    if not os.path.isdir(local):
        print("Restoring checkpoint:", latest)
        shutil.copytree(latest, local)
    print("Resuming from checkpoint:", local)
    return local


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. Run this on Saturn with the GPU accelerator enabled (Settings > Accelerator > GPU).")
    print("GPU:", torch.cuda.get_device_name(0))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("Loading base model", MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=CHECKPOINTING)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading data", DATA)
    ds = load_dataset("json", data_files=DATA, split="train")
    print("examples:", len(ds))

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    args = TrainingArguments(
        output_dir="./gptj-iraqi-out",
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        learning_rate=2e-4,
        num_train_epochs=EPOCHS,
        logging_steps=20,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=CHECKPOINTING,
        group_by_length=True,
        report_to=[],
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=256,
        packing=False,
        peft_config=lora,
        tokenizer=tokenizer,
        callbacks=[DriveCheckpointCallback()],
    )
    resume = find_resume_checkpoint()
    print("Training..." + (" (resuming)" if resume else ""))
    trainer.train(resume_from_checkpoint=resume)

    print("Saving adapter to", OUT)
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)

    print("Adapter saved in", OUT, "- save a notebook version to keep it")

    # ---- quick test: load the adapter and ask 3 Iraqi questions ----
    print("Testing the fine-tuned model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    tuned = PeftModel.from_pretrained(base, OUT)
    tuned.eval()
    questions = [
        "شلونك؟",
        "سويلي كود بايثون يطبع هلا",
        "شنو حلقة تكرار؟",
    ]
    for q in questions:
        prompt = f"السؤال: {q}\nالجواب:"
        inputs = tokenizer(prompt, return_tensors="pt").to(tuned.device)
        with torch.no_grad():
            out = tuned.generate(
                **inputs, max_new_tokens=60, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("=" * 40)
        print("Q:", q)
        print("A:", reply.strip())
    print("DONE. Adapter saved in", OUT)


if __name__ == "__main__":
    main()
