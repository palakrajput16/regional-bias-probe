# Regional Bias Probe: English and Hindi

A small, reproducible probe of how multilingual masked language models associate Indian regional identities (states) with positive and negative personality words, in English and in Hindi.

**Purpose:** measuring model bias, not making claims about real people. The test sentences contain stereotype-style statements only as probe inputs.

## Questions
1. Do XLM-R, mBERT and MuRIL associate some regional identities more with negative words than others?
2. Do English and Hindi sentences give the same picture?
3. Are the results stable across sentence templates and across models?

## Method
- 12 regional identities x 20 attribute words (10 positive, 10 negative) x 2 templates x 2 languages = 960 sentences per model
- For each sentence, mask the attribute word piece by piece and sum the log-probability the model gives to the original pieces
- For each attribute, subtract the average over identities (removes the effect of word frequency)
- Negative association of an identity = mean (negative attributes) minus mean (positive attributes), after that subtraction
- 95% ranges from resampling attributes 1000 times
- Checks: English vs Hindi agreement, template robustness, agreement between models (Spearman correlation)

## Results
<!-- TODO: paste results/association.png and the tables from results/correlations.csv -->

## What I found
<!-- TODO: 3 to 5 sentences in your own words. Mention what is stable and what is not. -->

## Limitations
- Only 12 identities, 20 attribute words and 2 templates per language
- Hindi sentences were written by hand by one person (me)
- Word-probability probing is a weak proxy; it shows associations learned from text, not truth about people
- Caste and religion are deliberately not included here
- No claims about downstream harm
<!-- TODO: add anything else you noticed -->

## Status
<!-- TODO: say honestly what is finished and what is in progress -->

## Reproduce
```
pip install -r requirements.txt
python -m src.run_probe --small     # quick test
python -m src.run_probe             # full run
python -m src.analyze
```
No dataset download or login is needed. All models are public on Hugging Face.

## Credits
Models: XLM-RoBERTa (FacebookAI), multilingual BERT (Google), MuRIL (Google, Apache-2.0).
