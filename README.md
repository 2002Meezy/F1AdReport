# 🏎️ F1 Ad Report — Análise de Visibilidade de Marcas

Sistema automatizado para detectar e analisar a visibilidade de patrocinadores em vídeos de corridas de Fórmula 1 usando deep learning (YOLO) e análise com inteligência artificial.

O sistema identifica logotipos de marcas em transmissões de F1, calcula métricas de visibilidade (tempo de exibição, frequência, confiança) e gera um **relatório PDF profissional** com gráficos e análise de IA.

---

## 📋 O que o sistema faz

1. **Detecta marcas** em vídeos de F1 usando YOLO11 (detecção de objetos)
2. **Rastreia métricas** por marca: tempo de exibição, contagem de frames, detecções, confiança
3. **Gera um vídeo anotado** com bounding boxes desenhadas em cada detecção
4. **Cria um relatório PDF** com tabelas, ranking Top 5 e gráficos (barras, pizza, confiança)
5. **Análise de IA** (opcional) usando um LLM local (Qwen via LM Studio)

### Marcas detectadas

AWS, American Express, Aramco, Cripto.com, DHL, ETIHAD, Ferrari, Globant, Haas, Heineken, KitKat, LV, Lenovo, MSC Cruises, Mercedes, Paramount+, Pirelli, Qatar Airways, Rolex, Salesforce, TAG Heuer, Vegas, Santander

---

## 🚀 Como usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Colocar o vídeo na pasta `input/`

```
Visual/
  input/
    seu_video.mp4     ← coloque aqui
```

Formatos suportados: `.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`

### 3. Rodar a análise

```bash
python main.py --weights runs/train/y11s-f1ads-v52/weights/best.pt --no-agent
```

### 4. Pegar os resultados na pasta `output/`

```
Visual/
  output/
    seu_video_analyzed.avi     ← vídeo com as detecções
    seu_video_report.pdf       ← relatório PDF com gráficos
```

---

## 📊 Relatório PDF

O sistema gera automaticamente um relatório PDF profissional com toda a análise de visibilidade das marcas no vídeo.

<p align="center">
  <img src="assets/report_preview.gif" alt="Preview do Relatório PDF" width="600"/>
</p>

### O que o relatório traz:

📄 **Página 1 — Resumo e Dados**
- **Cabeçalho** com nome do vídeo e data da análise
- **Cards de resumo** com duração, FPS, total de frames e quantidade de marcas detectadas
- **Tabela completa** com todas as marcas encontradas, ordenadas por tempo de exibição, mostrando: tempo visível (segundos), frames, total de detecções, porcentagem do vídeo e confiança média
- **Ranking Top 5** — as 5 marcas mais visíveis com barras de progresso comparativas e estatísticas detalhadas

📈 **Página 2 — Gráficos**
- **Gráfico de barras** mostrando o tempo de exibição de cada marca em segundos, com gradiente de cores
- **Gráfico de pizza (donut)** visualizando o market share de tempo de tela entre as marcas principais

🤖 **Página 3 — Análise de IA** *(opcional)*
- Análise escrita automaticamente por um modelo de linguagem (Qwen via LM Studio), com insights sobre a estratégia de posicionamento dos patrocinadores

---

## 🤖 Análise com IA (opcional)

Para incluir insights escritos por IA no relatório, rode o [LM Studio](https://lmstudio.ai/) localmente com um modelo carregado (ex: Qwen 4B):

```bash
# Com análise de IA
python main.py --weights runs/train/y11s-f1ads-v52/weights/best.pt

# Sem análise de IA (mais rápido)
python main.py --weights runs/train/y11s-f1ads-v52/weights/best.pt --no-agent
```

---

## ⚙️ Opções do Comando

```bash
python main.py --help
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--weights` | *(obrigatório)* | Caminho para o modelo YOLO (.pt) |
| `--source` | todos em `input/` | Vídeo específico para analisar |
| `--conf` | 0.25 | Threshold de confiança (0.0–1.0) |
| `--imgsz` | 640 | Tamanho da imagem na inferência |
| `--device` | 0 | `0` para GPU, `cpu` para CPU |
| `--input_dir` | input | Pasta de entrada |
| `--output_dir` | output | Pasta de saída |
| `--no-agent` | — | Pular análise de IA |
| `--lm-studio-url` | http://26.198.160.131:1234/v1 | URL da API do LM Studio |

---

## 📁 Estrutura do Projeto

```
Visual/
├── input/                  ← Coloque seus vídeos aqui
├── output/                 ← Resultados (vídeo anotado + PDF)
├── configs/                ← Configurações do dataset
│
├── main.py                 ← Ponto de entrada — pipeline completo
├── detect.py               ← Detecção YOLO standalone
├── brand_tracker.py        ← Rastreamento de métricas por marca
├── pdf_report.py           ← Gerador de PDF com gráficos matplotlib
├── ai_agent.py             ← Agente LangChain para análise com IA
│
├── train_basic.py          ← Treino básico do YOLO (parâmetros padrão)
├── train_advanced.py       ← Treino avançado (augmentation, cosine LR)
├── auto_label.py           ← Auto-anotação com Qwen3 VL (LM Studio)
│
└── requirements.txt        ← Dependências Python
```

---

## 🏋️ Treinar o Modelo

```bash
# Treino básico (rápido)
python train_basic.py

# Treino avançado (melhor acurácia)
python train_advanced.py --data configs/data2.yaml --epochs 150
```

---

## 📦 Requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (recomendado)
- [LM Studio](https://lmstudio.ai/) (opcional, para análise de IA)
