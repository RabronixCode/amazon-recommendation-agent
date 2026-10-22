# Amazon Electronics Recommender

A recommender system built and evaluated on Amazon Electronics reviews. The project compares rating-prediction baselines, a bias model, matrix factorization, item-based and user-based kNN, and a hybrid of matrix factorization and Item-kNN. It evaluates them on two different tasks (rating prediction and Top-N ranking) and ships a command-line agent that returns Top-N recommendations with a short explanation for each item.

Everything runs locally on a single CSV-like review file. Models are implemented from scratch with NumPy and pandas (scikit-learn is used only for metrics and cosine similarity).

## Main findings

1. **The best rating predictor is not the best ranker.** Matrix factorization has the lowest test RMSE (0.968), but at Top-N ranking it falls below the plain bias model (NDCG@10 0.132 versus 0.185). This agrees with the known observation that error metrics and top-N accuracy can disagree (Cremonesi, Koren and Turrin, 2010).
2. **Neighborhood models are strongest at ranking on this data.** Item-kNN reaches NDCG@10 0.212 and the Hybrid (0.25 MF + 0.75 Item-kNN) reaches 0.222, the best result in the project.
3. **The Bias versus MF ordering is stable.** Across 12 filtering thresholds (min user 3 to 5, min item 2 to 5), Bias beat MF (k=32 in that sweep) at ranking in every setting.
4. **Ties matter.** Several models produce many tied scores (GlobalMean and UserMean tie on all 100 candidates). Ranks are therefore computed with random tie-breaking inside the tied block, otherwise these baselines look better or worse than they really are.

These are results on one dataset, one split and one random draw of negatives. See [Limitations](#limitations) before drawing conclusions.

## Results

Test set, leave-one-out split, 14,956 users. Ranking metrics use 1 positive and 99 sampled negatives per user at k = 10. A random ranking scores about 0.10 HR@10 and about 0.045 NDCG@10 under this protocol.

| Model | HR@10 | NDCG@10 | MRR@10 | RMSE | MAE |
|---|---|---|---|---|---|
| GlobalMean | 0.0981 | 0.0441 | 0.0281 | 1.0571 | 0.8201 |
| UserMean | 0.0981 | 0.0441 | 0.0281 | 1.1284 | 0.7000 |
| ItemMean | 0.0314 | 0.0144 | 0.0093 | 1.0621 | 0.7469 |
| Bias (mu + bu + bi) | 0.3209 | 0.1847 | 0.1428 | 0.9945 | 0.7229 |
| MF (biased SGD, k=64) | 0.2551 | 0.1324 | 0.0953 | **0.9681** | 0.6565 |
| User-kNN (cosine, K=50) | 0.3214 | 0.1877 | 0.1467 | 1.0122 | 0.7277 |
| Item-kNN (adjusted cosine, K=50) | **0.3877** | 0.2124 | 0.1586 | n/a | n/a |
| Hybrid (alpha=0.25 MF + Item-kNN) | 0.3688 | **0.2217** | **0.1764** | 0.9716 | **0.6498** |

Item-kNN rating metrics were not computed. The Hybrid wins on NDCG and MRR while Item-kNN has the higher HR@10, which means the Hybrid places the positive item higher when it is found, but finds it in the top 10 slightly less often.

GlobalMean and UserMean give identical rankings because a user-specific constant does not change the order of candidates (their tie rate is 100%). Item-kNN has tied scores in about 20% of test cases, Bias in about 1%, and MF essentially never (`docs/tables/tie_rate_test_table.csv`). ItemMean is below random: ranking by raw item mean rating prefers items with few, very high ratings, which are rarely the held-out item. I did not test this explanation directly.

The Hybrid weight was selected by NDCG@10 on a random sample of 500 validation users (MF k=64 and Item-kNN K=50 fixed). The sweep is saved in `docs/tables/hybrid_alpha_sweep.csv` and plotted in `docs/figures/hybrid_alpha_sweep.png`. Validation NDCG@10 was 0.133 for pure MF (alpha=1), 0.248 for pure Item-kNN (alpha=0) and 0.271 for alpha=0.25, so the blend is better than either component on this sample.

### Sensitivity to filtering thresholds

The pipeline iteratively removes users and items with too few interactions. The table shows how the choice affects dataset size and the Bias and MF ranking results (test NDCG@10). Values with min user = 2 are missing because a user with two interactions cannot have train, validation and test items at the same time, so the pipeline stops at the split check.

| min user | min item | Interactions | Users | Items | Bias NDCG@10 | MF NDCG@10 |
|---|---|---|---|---|---|---|
| 3 | 2 | 147,299 | 36,690 | 7,074 | 0.1766 | 0.1190 |
| 3 | 3 | 142,632 | 35,694 | 5,712 | 0.1574 | 0.0992 |
| 3 | 4 | 137,694 | 34,619 | 4,766 | 0.1420 | 0.0947 |
| 3 | 5 | 132,725 | 33,541 | 4,048 | 0.1292 | 0.0922 |
| **4** | **2** | **81,104** | **14,956** | **5,793** | **0.1847** | **0.1333** |
| 4 | 3 | 75,751 | 14,078 | 4,400 | 0.1644 | 0.1085 |
| 4 | 4 | 69,930 | 13,122 | 3,383 | 0.1471 | 0.0951 |
| 4 | 5 | 64,493 | 12,198 | 2,695 | 0.1315 | 0.0865 |
| 5 | 2 | 48,767 | 7,089 | 4,739 | 0.1802 | 0.1216 |
| 5 | 3 | 42,857 | 6,317 | 3,267 | 0.1595 | 0.1050 |
| 5 | 4 | 36,932 | 5,508 | 2,322 | 0.1375 | 0.0881 |
| 5 | 5 | 31,521 | 4,761 | 1,689 | 0.1154 | 0.0789 |

NDCG values are not comparable between rows. The MF column in this sweep uses 32 factors (the pipeline above uses 64), so the 4/2 row differs slightly from the main table. Each row has a different set of users, items and therefore a different test set and a different pool of negatives. Only the relative order of models inside one row is meaningful. The thresholds used by the project (4 and 2, in bold) were fixed on dataset size and density considerations and should not be read as the setting with the best test score, because choosing thresholds by test metric would leak test information into the design. The sweep covers baselines, Bias and MF only. It is reproduced with `python experiments.py --threshold-sweep`, which writes one folder per setting under `docs/figures/threshold_sweep/` and `docs/tables/threshold_sweep/` plus `threshold_sweep_summary.csv`.

## Dataset and preprocessing

The input is an Amazon Electronics review dump in JSON lines format (optionally gzipped) with the fields `reviewerID`, `asin`, `overall` and `unixReviewTime`, as in the Amazon Review Data 2018 release (Ni, Li and McAuley, 2019). The file used here is a subset of Electronics reviews that was prepared by the author before this project, so it is not the full public dump, and the preparation steps are not part of this repository.

1. Rows are parsed line by line, tolerating malformed lines (a fallback parser handles the older Python-literal format).
2. Columns are renamed to `user_id`, `item_id`, `rating`, `timestamp`; nulls and exact duplicate rows are dropped.
3. Users with fewer than 4 interactions and items with fewer than 2 are removed iteratively until the set is stable.
4. Users and items are re-indexed to contiguous integers.

| | Interactions | Users | Items | Density |
|---|---|---|---|---|
| Before filtering | 915,003 | 728,125 | 11,257 | 0.011% |
| After filtering | 81,104 | 14,956 | 5,793 | 0.094% |

## Methodology

**Split.** Leave-one-out by time for each user: the latest interaction is the test item, the one before it is the validation item and the rest is training. This follows the common protocol of He et al. (2017).

**Models.**

| Model | Description |
|---|---|
| Baselines | Global, user and item mean ratings from the training set. |
| Bias | Prediction mu + bu + bi, fitted by coordinate descent with L2 regularization (reg 10, 20 iterations). |
| MF | Biased matrix factorization trained by SGD (Koren, Bell and Volinsky, 2009): 64 factors, learning rate 0.02, regularization 0.02, 20 epochs. The epoch with the lowest validation RMSE is kept. |
| Item-kNN | Adjusted cosine similarity between item vectors of mean-centered ratings (Sarwar et al., 2001), top 50 neighbors per item, weighted deviation prediction, falling back to the Bias prediction when the user has rated none of the neighbors. |
| User-kNN | Cosine similarity between mean-centered user vectors, top 50 neighbors, same fallback. |
| Hybrid | alpha * MF + (1 - alpha) * Item-kNN, with alpha chosen on validation (500-user sample). |

**Rating metrics.** RMSE and MAE on the validation and test items.

**Ranking metrics.** For each user the held-out item is ranked among itself plus 99 items the user never interacted with, sampled uniformly. HR@10, NDCG@10 and MRR@10 are reported for a single relevant item. Any interaction counts as relevant, regardless of the rating given. Ties are resolved by a random position inside the tied block. The final comparison table is computed on all 14,956 users, while the Hybrid alpha sweep and the Hybrid grid use a fixed random sample of 500 validation users to keep the Item-kNN scoring affordable.

**Hyperparameters.** The MF setting (64 factors, learning rate 0.02, regularization 0.02, 20 epochs) is the winner of `run_mf_grid_search`, which tries 36 configurations (factors 16/32/64, learning rate 0.01/0.02, regularization 0.02/0.05/0.1, epochs 10/20) and picks the lowest validation RMSE; all results are in `docs/tables/mf_grid_results.csv`. The differences between the top configurations are small: the best validation RMSE is 0.9313 (64 factors), and 16 factors reach 0.9319 and 32 factors 0.9326. The previous setting with 32 factors had a slightly lower test RMSE (0.9644 versus 0.9681), so the move to 64 factors is a validation-based choice and not a measurable improvement. The Hybrid weight was chosen on validation NDCG@10. A broader Hybrid grid (MF factors, MF regularization, Item-kNN K and alpha, with 10-epoch MF) is included in the code; its best validation configuration (64 factors, reg 0.02, K=20, alpha=0.25, validation NDCG@10 0.272 on the 500-user sample) is very close to the configuration used in the table above (K=50, validation NDCG@10 0.271), which was kept.

## Limitations

1. **Sampled metrics.** Ranking against 99 random negatives is cheaper than ranking against all items, but it can change the order of models compared with full ranking (Krichene and Rendle, 2020). The absolute numbers should not be compared with results from other protocols.
2. **Per-user temporal split, not a global time cutoff.** Within each user the order is correct: the test item is the latest one and nothing from that user's future is used for training. The protocol is still not a global-time split, because the models are trained on the other users' interactions, and some of those happened later than a given user's test interaction. The model can therefore learn item popularity or trends that were only visible after the moment it is asked to predict. This is a property of the protocol, not an error in the implementation, and a global cutoff would be the stricter alternative (see Ji et al., 2020, on leakage in offline evaluation).
3. **No uncertainty estimates.** All results come from one run with fixed seeds (42 for validation, 43 for test). Differences of about 0.01 NDCG, such as Hybrid versus Item-kNN, may not be robust. A bootstrap over users and several negative-sampling seeds would be the next step.
4. **Sparse neighborhoods.** With a density of 0.09%, most users have few rated neighbors of a candidate item. The average number of valid Item-kNN neighbors per prediction (over all scored user and candidate pairs in the validation and test ranking evaluation) is 0.04 out of 50, and User-kNN has at least one usable neighbor for only 24% (test) to 29% (validation) of the held-out items. Neighborhood models therefore fall back to the Bias prediction most of the time, which should be kept in mind when reading their results.
5. **Model selection in the agent.** `agent_cli.py` picks the default model by test NDCG@10. For a demo this is acceptable, but it means the reported test score of that model is optimistic.
6. **Implicit relevance.** A one-star review counts as a positive example in the ranking evaluation.
7. **Training speed.** SGD and kNN prediction loops are written in plain Python for clarity, so the full grid searches are slow.

## Repository structure

```
src/
  config.py        paths, required columns, default filtering thresholds, custom errors
  dataprep.py      loading, cleaning, filtering, re-indexing, leave-one-out split
  models.py        Bias, MF, Item-kNN and User-kNN
  evaluation.py    rating metrics, negative sampling, tie-aware ranking, HR/NDCG/MRR
  plots.py         all figures
  experiments.py   full experiment pipeline (EDA, baselines, grids, comparison tables)
  agent_cli.py     recommendation agent (training, artifacts, Top-N with explanations)
  demo.py          picks representative users for demonstrating the agent
data/              put the review file here (not tracked by git)
data_sample/       small example file in the expected format
docs/figures/      all figures produced by the experiments (generated, not tracked by git)
docs/tables/       all tables produced by the experiments, as CSV (generated, not tracked by git)
```

## Setup and usage

Requires Python 3 and the packages in `requirements.txt` (developed and checked with Python 3.13).

```
pip install -r requirements.txt
```

Place exactly one review file (`.json`, `.jsonl`, `.json.gz` or `.jsonl.gz`) in `data/`, or point the `RAW_REVIEWS_PATH` environment variable to it. Run everything from the `src` folder.

### Input data format

The full dataset is not part of the repository (`data/` is ignored by git), so you need to provide a review file yourself, or use the small example in `data_sample/` (see below). The loader expects:

- exactly one file in `data/` (or any path given in `RAW_REVIEWS_PATH`), with the extension `.json`, `.jsonl`, `.json.gz` or `.jsonl.gz`;
- one review per line, each line a JSON object (Python dict literals with single quotes are also accepted, as in the oldest Amazon dumps);
- at least these four fields per line, any other fields are ignored:

| Field | Type | Meaning |
|---|---|---|
| `reviewerID` | string | user identifier |
| `asin` | string | item identifier |
| `overall` | number | rating, 1 to 5 |
| `unixReviewTime` | integer | review time as Unix seconds |

Example line:

```
{"reviewerID": "A1B2C3", "asin": "B000123456", "overall": 5.0, "unixReviewTime": 1405123200, "reviewText": "..."}
```

Malformed lines are skipped with a warning, rows with missing values and exact duplicates are dropped. The Amazon Review Data (2018) release by McAuley's group, which has this schema, can be used (see reference 1; check the dataset page for the current download links). Note that the default thresholds (4 interactions per user, 2 per item) and the plain Python training loops were developed on a file of about 900,000 reviews. A much larger file will need a lot of memory and time, so take a subset first. ### Sample data

`data_sample/electronics_sample.jsonl` is a small example in exactly this format: 5,815 reviews from 1,188 users and 553 items, with only the four required fields (no review text and no reviewer names). It was cut from the prepared Electronics file used in this project by taking the 600 most reviewed items, drawing 1,200 random users with at least 4 reviews of those items, and applying the same iterative filtering (4 and 2). It is therefore much denser than the full data (0.89% against 0.09%) and is only meant to show the expected format and to let you run the code end to end in a few minutes. Results on it are not comparable with the tables in this README. The reviews come from the public Amazon Review Data; check its terms of use before redistributing more of it.

To run on the sample without copying it into `data/`:

```
# Linux / macOS
RAW_REVIEWS_PATH=../data_sample/electronics_sample.jsonl python experiments.py --no-hybrid-grid

# Windows PowerShell
$env:RAW_REVIEWS_PATH = "..\data_sample\electronics_sample.jsonl"; python experiments.py --no-hybrid-grid
```

Run the full experiment pipeline. Figures are written to `docs/figures/` and tables to `docs/tables/`, which are created automatically and ignored by git, so they appear only after you run the experiments:

```
python experiments.py
```

The Hybrid grid search runs by default and takes a long time (`--no-hybrid-grid` skips it). The MF grid search is off by default; `--mf-grid` runs it, trains 36 MF configurations, picks the lowest validation RMSE, writes `mf_grid_results.csv` and uses the selected model in all later steps, so labels and ranking results can change. The filtering-threshold sensitivity study is a separate mode:

```
python experiments.py --no-hybrid-grid       # main pipeline without the slow Hybrid grid
python experiments.py --mf-grid              # main pipeline with the MF grid search
python experiments.py --threshold-sweep      # threshold study instead of the main pipeline
```

Train and use the agent:

```
python agent_cli.py --retrain --demo                  # train, save artifacts, show examples
python agent_cli.py --user <reviewerID> --topn 10     # recommendations for one user
python agent_cli.py --user <reviewerID> --model Hybrid --alpha 0.25
python demo.py --with-raw                             # find users that illustrate each agent path
```

Users that were not in the training data receive a cold-start recommendation based on the global mean and item bias. For known users the agent returns the Top-N unseen items together with a model-specific explanation, for example the neighbor items that contributed to an Item-kNN score.

## References

1. Ni, J., Li, J. and McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP.
2. Koren, Y., Bell, R. and Volinsky, C. (2009). Matrix factorization techniques for recommender systems. IEEE Computer, 42(8).
3. Sarwar, B., Karypis, G., Konstan, J. and Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. WWW.
4. He, X., Liao, L., Zhang, H., Nie, L., Hu, X. and Chua, T.-S. (2017). Neural collaborative filtering. WWW.
5. Cremonesi, P., Koren, Y. and Turrin, R. (2010). Performance of recommender algorithms on top-N recommendation tasks. RecSys.
6. Krichene, W. and Rendle, S. (2020). On sampled metrics for item recommendation. KDD.
7. Ji, Y., Sun, A., Zhang, J. and Li, C. (2020). A critical study on data leakage in recommender system offline evaluation. arXiv:2010.11060.
