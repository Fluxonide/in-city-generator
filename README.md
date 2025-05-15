# Indian City Name Generator

Generate realistic Indian city names using a neural network trained on actual Indian cities.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Scrape the Indian city data:
```bash
python scrape.py
```

3. Train the model:
```bash
python train_full.py
```

## How it Works

The model uses a simple 3-layer MLP (Multi-Layer Perceptron) to learn patterns in Indian city names. It's trained on a dataset of real Indian cities and can generate new, realistic-sounding city names.

## Model Architecture

- Character-level language model
- 3-layer MLP with ReLU activation
- Trained on actual Indian city names
- Uses a context window of 12 characters
- Includes both English and Indian language characters

## Usage

After training, the model will automatically generate 40 example city names. You can modify the `train_full.py` script to generate more or fewer examples.

## Data Source

The training data is scraped from Wikipedia's list of cities in India by population. 