import pickle
#from src.utils import load_object

with open("C:/Users/Aeesha/DissProject/mlbearing/artifacts/model_20230922_041426.pkl", "rb") as file:
    model = pickle.load(file)

print('md', model)
print(type(model))
