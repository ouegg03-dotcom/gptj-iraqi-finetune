#!/bin/bash
# GPT-J QLoRA training driver for the Saturn Cloud job.
# Files come from the cloned repo in ./repo. The training script expects
# training_data.jsonl in the working directory and writes ./gptj-iraqi-lora,
# ./gptj-iraqi-checkpoints and ./gptj-iraqi-out there.
cd /home/jovyan/workspace || exit 1

echo "===== ENVIRONMENT ====="
nvidia-smi || true
df -h /home/jovyan/workspace || true
free -g || true
python3 --version
command -v curl; command -v python3

echo "===== FILES ====="
ls -la repo
cp repo/training_data_v1_9640.jsonl training_data.jsonl
wc -l training_data.jsonl
ls -la

echo "===== CONNECTIVITY ====="
for url in https://pypi.org/simple/ https://github.com https://huggingface.co https://raw.githubusercontent.com https://catbox.moe https://0x0.st; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -I "$url" || echo FAIL)
  echo "$url -> $code"
done

upload_file() {
  local file="$1"
  local url=""
  url=$(curl -s -F "fileToUpload=@$file" -F "reqtype=fileupload" --max-time 900 https://catbox.moe/user/api.php) || url=""
  if [ -n "$url" ] && [ "${url:0:4}" = "http" ]; then echo "$url"; return 0; fi
  url=$(curl -s -F "file=@$file" --max-time 900 https://0x0.st) || url=""
  if [ -n "$url" ] && [ "${url:0:4}" = "http" ]; then echo "$url"; return 0; fi
  echo "UPLOAD FAILED for $file"
  return 1
}

# Background checkpoint uploader: first run after 30 min, then every 30 min.
(
  sleep 1800
  while true; do
    if [ -d /home/jovyan/workspace/gptj-iraqi-checkpoints ]; then
      cd /home/jovyan/workspace
      rm -f checkpoints.zip
      python3 -m zipfile -c checkpoints.zip gptj-iraqi-checkpoints
      if [ -f checkpoints.zip ]; then
        echo "CHECKPOINT UPLOAD ($(date), $(du -h checkpoints.zip | cut -f1)):"
        upload_file checkpoints.zip || true
        rm -f checkpoints.zip
      fi
    fi
    sleep 1800
  done
) &
UPLOADER=$!

echo "===== TRAINING START ====="
python3 repo/finetune_gptj_saturn.py 2>&1 | tee train.log
TRAIN_EXIT=${PIPESTATUS[0]}
kill $UPLOADER 2>/dev/null || true
echo "===== TRAINING FINISHED (exit $TRAIN_EXIT) ====="

echo "===== OUTPUTS ====="
ls -la gptj-iraqi-lora 2>/dev/null || echo "NO ADAPTER DIR"
ls -la gptj-iraqi-checkpoints 2>/dev/null || echo "NO CHECKPOINT DIR"
du -sh gptj-iraqi-lora gptj-iraqi-checkpoints 2>/dev/null || true

echo "===== UPLOADING ADAPTER ====="
if [ -d gptj-iraqi-lora ]; then
  rm -f adapter.zip
  python3 -m zipfile -c adapter.zip gptj-iraqi-lora
  echo "ADAPTER URL:"
  upload_file adapter.zip || true
fi
if [ -d gptj-iraqi-checkpoints ]; then
  rm -f checkpoints-final.zip
  python3 -m zipfile -c checkpoints-final.zip gptj-iraqi-checkpoints
  echo "CHECKPOINTS URL:"
  upload_file checkpoints-final.zip || true
fi
echo "===== ALL DONE (train exit $TRAIN_EXIT) ====="
exit "$TRAIN_EXIT"
