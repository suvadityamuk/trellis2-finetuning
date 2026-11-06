# trellis2-prep: any folder of 3D meshes -> TRELLIS.2 latents + renders on the Hub.
# Published as cr.hf.co/suvadityamuk/trellis2-finetuning-prep: the base image plus the
# files below (a no-op over a base built from container/Dockerfile, which has them).
#   docker build -f container/prep.Dockerfile -t trellis2-finetuning-prep .
ARG BASE=cr.hf.co/suvadityamuk/trellis2-finetuning:latest
FROM ${BASE}
COPY container/trellis2/data_toolkit/datasets/MeshFolder.py /workspace/TRELLIS.2/data_toolkit/datasets/MeshFolder.py
COPY container/trellis2/byod/prep.py /workspace/TRELLIS.2/byod/prep.py
COPY --chmod=755 container/trellis2/byod/bin/trellis2-prep /usr/local/bin/trellis2-prep
CMD ["trellis2-prep", "--help"]
