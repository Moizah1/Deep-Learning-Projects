# Review Sentiment Analyzer

A Jupyter notebook that classifies social-media reviews and comments as **negative**, **neutral** or **positive** using a pre-trained transformer, then summarises the results by platform and over time.

It uses [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model trained on tweets. It handles short, informal text with emoji, slang and ALL-CAPS emphasis, and its neutral class means comments like "meh" or "it's fine" aren't forced into positive or negative.
