import os
import joblib

model_paths = [
    ("Random Forest", "models/random_forest.pkl"),
    ("SVM", "models/svm.pkl")
]

for name, path in model_paths:
    print("\nModel:", name)

    if os.path.exists(path):
        model = joblib.load(path)

        if hasattr(model, "classes_"):
            classes = model.classes_
        elif hasattr(model, "named_steps"):
            classifier = list(model.named_steps.values())[-1]
            classes = classifier.classes_
        else:
            print("Cannot find model categories.")
            continue

        print("Categories:", list(classes))
        print("Total categories:", len(classes))

    else:
        print("Model file not found:", path)

print("\nCompleted!")