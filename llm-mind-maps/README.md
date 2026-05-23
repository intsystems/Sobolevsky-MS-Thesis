# Mind Map Generation Using Large Language Models
This is a proof-of-concept experiment that tests LLMs' ability to generate sentence-based mind maps. For details, check out the [project report](https://github.com/TeoSable/llm-mind-maps/report/report.pdf) in this [subproject's repository](https://github.com/TeoSable/llm-mind-maps).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TeoSable/llm-mind-maps
cd llm-mind-maps
```

2. (Recommended) Create a clean virtual environment for the project:
```bash
python3 -m venv venv
```

Activate the environment:

Linux/Mac:
```bash
source venv/bin/activate
```

Windows:
```
venv/bin/Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

It is highly recommended that you use a GPU for running the project, since it involves inferencing an LLM locally. Keep in mind that the default model for the experiment, Qwen 2.5-3B Instruct, requires at least 8 GB of GPU memory to run smoothly. You can check CUDA availability using:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If needed, visit the [official PyTorch website](https://pytorch.org/) for a guide on installation with CUDA enabled.

## Usage

For a quick run on three documents from the development subset:
```bash
python run.py \
  --data-dir data \
  --split dev \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-files 3
```

For the full test split experiment with 1-shot Qwen2.5-3B-Instruct:
```bash
python run.py \
  --data-dir data \
  --split test \
  --model Qwen/Qwen2.5-3B-Instruct \
  --few-shot-count 1 \
  --output-json outputs/qwen25_3b_test_1shot.json
```

For the full test split experiment with 1-shot Qwen3-4B-Instruct:
```bash
python run.py \
  --data-dir data \
  --split test \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --quantization 4bit \
  --few-shot-count 1 \
  --output-json outputs/qwen3_4b_test_1shot.json
```

For more information on command-line arguments of `run.py`, run the following command:
```bash
python run.py --help
```