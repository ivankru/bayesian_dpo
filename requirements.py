# Среда для bayesian_dpo: DPO, AlpacaEval
# Python >= 3.10

torch>=2.5.1
transformers>=4.40.0
peft>=0.10.0
datasets>=2.14.0
accelerate>=0.20.0
numpy>=1.24.4
tqdm
scikit-learn>=1.3.2
mlflow>=2.8.0

# Опционально, для официального alpaca_eval (--alpaca-eval-lib):
# alpaca-eval>=0.6.6

# Опционально, для ifeval_run.py (Google IFEval, скачиваемый в ~/.cache/soft_dpo/ifeval/):
# absl-py>=1.0.0
# langdetect>=1.0.9
# nltk>=3.8.1
# immutabledict>=2.0.0
