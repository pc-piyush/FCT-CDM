import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import yaml
from src.data_loader import OMOPLoader
from src.cohort import build_cohort
from src.tensor_builder import HealthTensorBuilder
from src.feature_matrix import build_feature_matrix
from src.labels import label_aki
from src.models import train_model
from src.evaluation import evaluate
from sklearn.model_selection import train_test_split

# load config
config = yaml.safe_load(open("config/config.yaml"))

# load data
loader = OMOPLoader(config)
data = loader.load_all()

# cohort
cohort, data = build_cohort(data, config)

# tensor
tensor_builder = HealthTensorBuilder(config)
tensors = tensor_builder.build_tensor(cohort, data)

# baseline features
X = build_feature_matrix(data)

# labels
y_dict = label_aki(data)

# align
X = X.loc[X.index.intersection(y_dict.keys())]
y = [y_dict[i] for i in X.index]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train
model = train_model(X_train, y_train)

# eval
results = evaluate(model, X_test, y_test)

print(results)