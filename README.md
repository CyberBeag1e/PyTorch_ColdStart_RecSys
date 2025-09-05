## Demo Recommendation System for Cold Start Problem with PyTorch

### Dataset
- [MovieLens](https://grouplens.org/datasets/movielens/). Use ml-latest-small (100,000 ratings and 3,600 tag applications applied to 9,000 movies by 600 users.) (default), or switch to ml-latest (approximately 33,000,000 ratings and 2,000,000 tag applications applied to 86,000 movies by 330,975 users.)
- [TMDB](https://www.themoviedb.org/?language=en-US) metadata: textual features (overview, title) + posters

To download MovieLens dataset and enrich them with TMDB metadata. Run the following pipeline.

```bash
python -m src.data_utils.download --version small
python -m src.data_utils.merge
```

Note that before merging ML dataset with TMDB metadata, you need to get a TMDB API Key, create a `.env` file and set `TMDB_API_KEY = YOUR_KEY`.

Also node that for path configuration, change directories and paths in `src.config.path_config.py`

### Cold Start Problem

To simulate NEW items (cold start for new items instead of new users), specified a global cutoff point (either by selecting a quantile or a date) and do temporal split:
- All items that released before the cutoff point are defined as SEEN items.
- All items that have no observed user-item interactions before the cutoff point among the SEEN items are defined as NEW items.
- Training data are the interactions with timestamp earlier than the cutoff point. Post-cutoff data are the interactions with timestamp no earlier than the cutoff point. The users with no records in the training data are dropped from the post-cutoff data.
- Among the post-cutoff data, the first small proportion of data are selected as validation data, and the rest of post-cutoff data are selected as test data.

To do the temporal split, run the following pipeline, and refer to the file docs for argument choices.
```bash
python -m src.data_utils.temporal_split
```

We first experiment with two baseline models
- Popularity-based recommendation: recommend top `k` items that the user has not interacted with based on popularity.
- Matrix factorization with BPR loss: recommend top `k` items that the user has not interacted with using matrix factorization with BPR loss. Use binary relevance (interacted or not) rather than ratings as targets.

Problems with baseline models: while they preform well on SEEN data, they both have ~0 Recall@k and ~0 nDCG@k, indicating they suffer from cold start problem.

To run baseline models, run the following pipelines, and refer to the file docs for argument choices.
```bash
python -m src.scripts.baseline_popularity
python -m src.scripts.baseline_mf_bpr
```

### Two Tower Model

- Item Tower: encode textual features (pre-computed text embeddings via SentenceBERT), posters (pre-computed image embeddings via CLIP, optional) and item id (small item id embeddings, optional) as item embeddings
- User Tower: encode `L` historically interacted items (items embeddings) as user embeddings, with attention pooling (by default) or mean pooling.
- Loss function: InfoNCE with in-batch loss - treat positive items of the other users as negatives to this user.

Results: nonzero Recall@k and nDCG@k on NEW items, slightly lower metrics on ALL and SEEN items compared to baseline models.

To run the two tower model, run the following pipeline, and refer to the file docs for argument choices.
```bash
python -m src.scripts.run_two_towers --use_text --use_img
```

Note that by default we forced the computation of text embeddings, image embeddings, and training of MF-BPR and Two Tower models on CUDA.

