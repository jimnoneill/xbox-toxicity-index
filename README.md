# Forecasting provocation and posting rhythm in the United States Congress

Analysis code for O'Neill, Cabanillas, Kepner, Goggin, and Brooks (2026), *Forecasting provocation and posting rhythm in the United States Congress* (manuscript in preparation for EPJ Data Science).

The code scores congressional social media posts with fourteen text classifiers, builds a seven-dimension Toxicity Index (TI7), computes posting-rhythm features on cleaned event streams, and runs a forecasting program with rolling origins, lagged-only features, proper scoring rules, and a final test window scored once.

## Contents

| Path | Purpose |
|---|---|
| `xbox/classifiers.py` | Fourteen-head scoring (six off-the-shelf, eight purpose-trained SetFit heads) |
| `xbox/hygiene.py` | Collection-outage detection, thread collapsing, scheduler signatures |
| `xbox/temporal_v2.py` | Rhythm descriptors: finite-size burstiness A_n, hour-of-week entropy, night and weekend shares, sessions, rapid follow-up |
| `xbox/location.py` | Member location per day from Voteview roll-call attendance |
| `xbox/hawkes.py`, `xbox/pointprocess_models.py` | Hawkes EM and 19 point-process variants with time-rescaling checks |
| `xbox/binned_model.py`, `xbox/binned_model_b.py` | Negative-binomial hourly count model with Congress-wide shocks, regimes, and partial pooling |
| `xbox/forecast_p1.py` | Next-post provocation from rhythm (P1) |
| `xbox/forecast_p2.py` | Volume and burst forecasting ladder (P2) |
| `xbox/forecast_p3.py` | Member-year trajectories and power-status features (P3) |
| `xbox/kistner_ext.py` | External replication on the Kistner et al. (2026) 2009 to 2022 panel |
| `scripts/v7/` | Entry points for every analysis and figure in the manuscript |
| `tests/` | Leakage perturbation tests and estimator checks |

Modules `compulsion_signatures.py`, `disorder_baseline.py`, and `fusion.py` belong to an earlier version of the project and are not used by the manuscript analyses.

## Data

Not included. The congressional corpus is the public congresstweets archive (github.com/alexlitel/congresstweets); roll-call data come from Voteview; the replication panel is Harvard Dataverse doi:10.7910/DVN/19MIBB. The case-study archives were collected through the X API and are not redistributed. Scripts that call the X API read a bearer token from a local credentials file outside the repository.

## Models

The eight purpose-trained classifiers are on Hugging Face: huggingface.co/jimnoneill/xbox-compulsion-classifier.

## Setup

```
pip install -r requirements.txt
python -m pytest tests
```

## License

Code: MIT. Classifier weights: CC BY-NC 4.0 (see the Hugging Face model card).
