import pickle

data = pickle.load(open("data/processed/tensors.pkl","rb"))

pid = list(data.keys())[0]

print("TYPE:", type(data[pid]))
print("SAMPLE:", data[pid][:10])