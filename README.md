# Pipeline de Detecção de Mosquitos

FoundIR (restauração) → compressão PNG→JPG → StableSR (super-resolução) → detecção de objetos (YOLOv26 ou RT-DETR).

**Requisitos:** Linux, Conda, GPU NVIDIA, ffmpeg ou Pillow.

## Instalação

```bash
git clone https://github.com/pantoja/mosquito-detection-pipeline pipeline && cd pipeline
```

Edite `config.sh` com os diretórios de destino do FoundIR e StableSR, depois execute:

```bash
./setup.sh
```

O setup clona os repositórios, cria os ambientes conda e oferece download automático dos pesos via `huggingface-cli`. Se aceitar, os caminhos são configurados automaticamente em `weights.env.sh`.

**Download manual:** coloque os arquivos em `weights/{foundir,stablesr,yolov26,rtdetr}/` e copie o peso do FoundIR para `$FOUNDIR_DIR/premodel/model-2000.pt`.

## Pesos

Hospedados em https://huggingface.co/vicpantoja2/aedes-egg-weights:

| Modelo | Pasta |
|--------|-------|
| FoundIR | `foundir/` |
| StableSR | `stablesr/` |
| RT-DETR | `rtdetr/` |
| YOLOv26 | `yolov26/` |

## Uso

```bash
./run.sh --input /caminho/para/imagens/ --detector yolo
./run.sh --input /caminho/para/imagem.png --detector rtdetr
```

Entrada: PNG (arquivo único ou pasta).

## Saída

```
output/YYYYMMDD_HHMMSS/
├── 01_foundir/            # PNG restaurados
├── 02_foundir_compressed/ # JPG comprimidos (qualidade 95)
├── 03_stablesr/           # imagens com super-resolução
├── 04_results/
│   ├── annotated/         # imagens com bounding boxes
│   └── data/              # JSON por imagem: class_id, class_name, confidence, bbox_xyxy
└── errors.log
```
