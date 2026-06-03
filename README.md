# Pipeline de Detecção de Mosquitos

Pipeline de quatro etapas: restauração com FoundIR → compressão PNG → super-resolução com StableSR → detecção de objetos.

## Requisitos

- Linux (testado em HPC com SLURM; executa sequencialmente em uma única máquina)
- Conda
- GPU NVIDIA com CUDA
- ffmpeg (ou Python com Pillow) para compressão

## Instalação

### 1. Clone este repositório

```bash
git clone https://github.com/pantoja/mosquito-detection-pipeline pipeline && cd pipeline
```

### 2. Edite o `config.sh`

Abra `config.sh` e defina apenas:
- `FOUNDIR_DIR` — onde clonar o FoundIR
- `STABLESR_DIR` — onde clonar o StableSR
- Nomes dos ambientes conda (os padrões já estão configurados)

Os caminhos dos pesos são detectados e configurados automaticamente pelo `setup.sh` após o download.

### 3. Execute o setup

```bash
./setup.sh
```

O script clona FoundIR e StableSR, cria três ambientes conda e oferece a opção de baixar os pesos automaticamente via `huggingface-cli`. Se optar pelo download automático, os caminhos dos pesos são configurados sem necessidade de edição manual.

### 4. Download manual dos pesos (opcional)

Se preferir baixar manualmente, coloque os arquivos nas pastas correspondentes dentro de `weights/` e edite os caminhos em `config.sh`. O peso do FoundIR deve ser copiado para:

```
$FOUNDIR_DIR/premodel/model-2000.pt
```

## Pesos

Todos os pesos estão hospedados em: https://huggingface.co/vicpantoja2/aedes-egg-weights

| Modelo | URL |
|--------|-----|
| FoundIR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/foundir |
| StableSR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/stablesr |
| RT-DETR | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/rtdetr |
| YOLOv26 | https://huggingface.co/vicpantoja2/aedes-egg-weights/tree/main/yolov26 |

Download via huggingface-cli:

```bash
huggingface-cli download vicpantoja2/aedes-egg-weights --local-dir ./weights
```

## Uso

```bash
# Executar com detecção YOLO
./run.sh --input /caminho/para/imagens/ --detector yolo

# Executar com detecção RT-DETR
./run.sh --input /caminho/para/imagem.png --detector rtdetr
```

A entrada deve ser PNG. Pode ser um único arquivo ou uma pasta com arquivos PNG.

## Saída

Cada execução cria um diretório com timestamp em `output/`:

```
output/YYYYMMDD_HHMMSS/
├── 01_foundir/               # imagens restauradas (PNG)
├── 02_foundir_compressed/    # imagens comprimidas (JPG, qualidade 95)
├── 03_stablesr/              # imagens com super-resolução
├── 04_results/
│   ├── annotated/            # imagens com bounding boxes
│   └── data/                 # arquivos JSON com detecções
└── errors.log                # falhas por imagem
```

Formato do JSON de detecção:

```json
{
  "image": "example.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "aedes",
      "confidence": 0.95,
      "bbox_xyxy": [120.5, 80.2, 340.1, 210.8]
    }
  ]
}
```
