from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from dataset_loader import load_dataset

def train_model(dataset_path):
    print(dataset_path)
    X, y = load_dataset(dataset_path)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(classifier, 'asl_svm_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model and label encoder saved successfully.")


