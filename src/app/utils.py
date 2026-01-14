import pickle

def load_model(name):
    with open(f"models/{name}.pkl", "rb") as f:
        return pickle.load(f)

def predict(model, X):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    return pred, proba
