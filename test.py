import pickle

fmap = pickle.load(open("data/artifacts/feature_map.pkl","rb"))
print("Feature count:", len(fmap))
print(list(fmap.keys())[:10])