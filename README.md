# site-mapper

## Setup

Modify settings in crawler.py then

```shell
uv run python -m site_mapper.crawler
```

## Settings

See the scope_rules dictionary at the bottom of crawler.py.

For quick testing, it's set to stop the crawl after 4 pages. You can change that by
modifying the `scope_rules["page_limit"]` key.

After the page_limit is reached the crawler will print a report of every page it 
visited, along with the outlinks it found on each of those pages.

## Training pipeline (quickstart)

This repo includes a small fixture so you can train and evaluate the Random Forest model without running a crawl.

### 1) Quickstart with fixture (no crawling)

1. Create results directory:
   
   ```shell
   mkdir results
   ```

2. Copy the sample dataset into the expected location:


   # Windows (cmd)
   copy tests\fixtures\training_data_v2.sample.csv results\training_data_v2.csv
   
   # macOS/Linux
   cp tests/fixtures/training_data_v2.sample.csv results/training_data_v2.csv


3. Preprocess features and save processed CSV:

   uv run python src/site_mapper/preprocess_for_ml.py


4. Train the Random Forest model and save artifacts to `results/model/`:

   uv run python src/site_mapper/train_random_forest.py

Artifacts written:
- `results/training_data_processed.csv`
- `results/model/random_forest_model.joblib`
- `results/model/model_info.json`

### 2) Full pipeline with crawler (optional)

If you want to regenerate the dataset from a tiny crawl (default `page_limit: 5` in `config.yaml`):

```shell
uv run pip install -e .
uv run python -m playwright install

# Run a small crawl
uv run python -m site_mapper.cli --url https://archive-it.org/explore --config config.yaml

# Build training data from crawl output
uv run python scripts/prepare_training_data_v2.py

# Preprocess and train
uv run python src/site_mapper/preprocess_for_ml.py
uv run python src/site_mapper/train_random_forest.py
```

### Notes
- The training script prefers the `label_fixed` column if present, otherwise uses `label_contextual` from the prepared data.
- For a focused review, this PR includes the training pipeline and a small dataset fixture. Crawler, demos, and analysis tools can be reviewed in follow-up PRs.