# Predicting-survey-responses-with-llms
Fine-tuned Falcon-RW-1B using QLoRA to predict Indonesian public opinion survey responses.

This repository contains code and data for replicating the fine-tuning of a large language model (Falcon-rw-1b) on Indonesian World Values Survey (WVS) data. This work investigates the ability of large language models (LLMs) to simulate human survey responses.

## Files

- **WVS Indonesia Survey Data Processing for LLM Prediction.R**  
  R script for preprocessing WVS Time Series (1981–2022) data and generating prompt–response pairs.

- **wvs_prompt_response_indonesia.csv**  
  Processed dataset used as input for fine-tuning. Each row contains a structured prompt and the corresponding survey answer.

- **Model Fine-tuning replication code.py**  
  Python script for formatting prompts and fine-tuning Falcon-rw-1b using QLoRA.
  
- **README.md**  
  This file.


## Citation

If you use this code or data in your work, please cite:

> Park, Jiyoung (2025). *Can AI Predict the Future of Public Opinion? An Empirical Test with Indonesian Respondents*. [Working Paper].

