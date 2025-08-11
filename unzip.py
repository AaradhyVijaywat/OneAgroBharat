import gzip
import pickle

# Decompress and load the model from .gz
model_gz_path = "model.pkl.gz"
with gzip.open(model_gz_path, "rb") as f_in:
    model_obj = pickle.load(f_in)

# Save the model as an uncompressed .pkl
with open("model.pkl", "wb") as f_out:
    pickle.dump(model_obj, f_out)

print("Model saved uncompressed as model.pkl")

# Decompress and load the preprocessor from .gz
preprocessor_gz_path = "preprocessor.pkl.gz"
with gzip.open(preprocessor_gz_path, "rb") as f_in:
    preprocessor_obj = pickle.load(f_in)

# Save the preprocessor as an uncompressed .pkl
with open("preprocessor.pkl", "wb") as f_out:
    pickle.dump(preprocessor_obj, f_out)

print("Preprocessor saved uncompressed as preprocessor.pkl")