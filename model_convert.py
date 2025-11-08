import pickle
import joblib
import os

models_dir = "5_DjangoApp/facerecognition/static/models"

files = [
    "machinelearning_face_emotion.pkl",
    "machinelearning_face_person_identity.pkl"
]

for f in files:
    pkl_path = os.path.join(models_dir, f)
    model = pickle.load(open(pkl_path, "rb"))
    new_path = pkl_path.replace(".pkl", ".joblib")
    joblib.dump(model, new_path)
    print(f"✅ Converted {f} → {os.path.basename(new_path)}")
