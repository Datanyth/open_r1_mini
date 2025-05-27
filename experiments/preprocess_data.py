from datasets import load_dataset

dataset = load_dataset("ChaosAiVision/en_processed_open-s1")

dataset = dataset.remove_columns(['conversations'])
dataset.push_to_hub("presencesw/en_processed_open-s1", private=False)