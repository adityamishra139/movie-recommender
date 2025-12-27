import pickle
import os

class ModelPersistence:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

    def save(self, obj, filename):
        """
        Saves a Python object to disk using pickle.
        """
        path = os.path.join(self.artifact_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved: {path}")

    def load(self, filename):
        """
        Loads a Python object from disk using pickle.
        """
        path = os.path.join(self.artifact_dir, filename)
        with open(path, "rb") as f:
            return pickle.load(f)
