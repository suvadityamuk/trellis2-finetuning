# trellis2-train: config.json + trellis2-prep latents -> LoRA checkpoints on the Hub,
# metrics on Trackio. Published as cr.hf.co/suvadityamuk/trellis2-finetuning-train: the
# base image plus the files below (a no-op over a base built from container/Dockerfile).
#   docker build -f container/train.Dockerfile -t trellis2-finetuning-train .
ARG BASE=cr.hf.co/suvadityamuk/trellis2-finetuning:latest
FROM ${BASE}
COPY container/trellis2/finetune_lora.py /workspace/TRELLIS.2/finetune_lora.py
COPY container/trellis2/byod/train.py /workspace/TRELLIS.2/byod/train.py
COPY --chmod=755 container/trellis2/byod/bin/trellis2-train /usr/local/bin/trellis2-train
CMD ["trellis2-train", "--help"]
