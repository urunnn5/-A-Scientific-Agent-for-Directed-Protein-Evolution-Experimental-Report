import joblib


class MLPPredictor:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    @classmethod
    def load(cls, path):
        return cls(*joblib.load(path))

    def predict(self, sequences):
        x = self.encoder.transform([list(seq) for seq in sequences])
        return self.model.predict(x)
