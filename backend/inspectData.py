from datasets import load_dataset
ds = load_dataset("strova-ai/hr-policies-qa-dataset", split="train")
print(ds)
print(ds.column_names)
print(ds[0])