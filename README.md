# Emotion Classification with DistilBERT

Fine-tuned DistilBERT to classify short texts into 6 emotions (sadness, joy, love, anger, fear, surprise), benchmarked against a TF-IDF + Logistic Regression baseline.

**Live demo:** [add your Hugging Face Space link]
**Model:** [add your Hugging Face model link]

## Problem
[1-2 sentences: what is being classified, and who would use it?]

## Data
`dair-ai/emotion`: 16000 train / 2000 validation / 2000 test. The classes are imbalanced (see `results/eda.png`), so I report **macro-F1**, which weights every class equally, alongside accuracy.

## Approach
1. **Baseline:** TF-IDF (unigrams and bigrams) + Logistic Regression, with `C = 100` selected on the validation set.
2. **Main model:** `distilbert-base-uncased` fine-tuned for 3 epoch(s) (learning rate 2e-05, batch size 16, max length 64); best epoch chosen by validation macro-F1.
3. The test set was used once, for the final comparison.

## Results (test set)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.873 | 0.810 |
| DistilBERT (fine-tuned) | 0.926 | 0.882 |

![Confusion matrices](results/confusion_matrices_test.png)
![Per-class F1](results/per_class_f1.png)

## Error analysis
[4-5 sentences from your hand-labeled errors: which classes are confused, how many errors were ambiguous, mislabeled, sarcastic or genuine model mistakes, and what you would try next.]

## How to run
```bash
pip install -r requirements.txt
jupyter notebook emotion_classifier.ipynb   # use Google Colab (GPU) for the training step
python app/app.py                           # launch the demo locally
```

## Limitations and next steps
- [e.g. English only, short texts; labels may be noisy]
- [e.g. try a larger model, class weights for rare classes, more data]

## Tech stack
Python, PyTorch, Hugging Face Transformers and Datasets, scikit-learn, pandas, Gradio
