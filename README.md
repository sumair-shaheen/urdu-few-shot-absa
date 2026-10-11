Few-Shot Aspect-Based Sentiment Analysis for Low-Resource Urdu Text
Project Overview
This project investigates few-shot Aspect-Based Sentiment Analysis (ABSA) for low-resource Urdu text using transformer models.
The goal is to evaluate how different transformer architectures perform when only a limited amount of labeled training data is available. The project also explores full fine-tuning and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA).
Research Objectives
•	Investigate few-shot learning for Urdu ABSA.
•	Evaluate transformer models under different supervision levels.
•	Compare full fine-tuning with LoRA-based parameter-efficient fine-tuning.
•	Analyze model performance using sentiment classification metrics.
Technologies
•	Python
•	PyTorch
•	Hugging Face Transformers
•	Pandas and NumPy
•	Scikit-learn
•	PEFT (Parameter-Efficient Fine-Tuning)
Project Structure
•	config.py — Project configuration and file paths.
•	requirements.txt — Python dependencies.
•	01_check_environment.py – 13_ate_finetune.py — Data preparation, model training, evaluation, and related experimental scripts.
•	urdu_few_shot_absa.ipynb — Jupyter notebook for project experiments.
Data and Model Availability
Datasets, trained model weights, and other large experimental files may be excluded from this repository because of their size, licensing, or sharing restrictions.
Research Status
This repository supports research on few-shot learning and aspect-based sentiment analysis for low-resource Urdu text. Experimental details and results should be interpreted in the context of the thesis methodology.

