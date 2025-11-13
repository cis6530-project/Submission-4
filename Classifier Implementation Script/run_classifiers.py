import os, re, glob, time
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# file parameters, pulling from code filepath 
DATA_DIR = os.path.abspath("processed_dfs")
EXPORT_DIR = os.path.abspath("Results")
os.makedirs(EXPORT_DIR, exist_ok=True)

# KNN rubric (k=3.5) - round to a valid integer
KNN_K = max(1, int(round(3.5)))  # -> 4

summary = [] # to output a global summary

def load_feature_set(gram):
    Xtr = pd.read_csv(os.path.join(DATA_DIR, f"X_train_{gram}g.csv"))
    Xte = pd.read_csv(os.path.join(DATA_DIR, f"X_test_{gram}g.csv"))
    ytr = pd.read_csv(os.path.join(DATA_DIR, f"y_train_{gram}g.csv"), header=None).squeeze() # required for training CSV file does not contain a header row
    yte = pd.read_csv(os.path.join(DATA_DIR, f"y_test_{gram}g.csv"), header=None).squeeze() # squeese for pandas convering to a 1D array for processing
    return Xtr, Xte, ytr, yte

def generate_totals(model_name, ngram_type, acc, prec, rec, f1):
    summary.append({
        "Model": model_name,
        "Feature": ngram_type,
        "Accuracy": round(acc*100, 2),
        "Precision": round(prec*100, 2),
        "Recall": round(rec*100, 2),
        "F1 Score": round(f1*100, 2)
    })

def main():
    # Load preprocessed data
    X_train_1g, X_test_1g, y_train_1g, y_test_1g = load_feature_set(1)
    X_train_2g, X_test_2g, y_train_2g, y_test_2g = load_feature_set(2)

    # process 1-GRAM opcodes -------------------------------------------------------------------
    # ---- 1-GRAM: Decision Tree ----
    print("\n--- Decision Tree (1-gram) ---")
    Xtr = X_train_1g.fillna(0.0).astype(float)
    Xte = X_test_1g.fillna(0.0).astype(float)
    #clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_impurity_decrease=0.001, random_state=0) # strong inital accuracy and fair for the small APT's
    #clf_dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, min_impurity_decrease=0.01, random_state=0) # more general and overfits
    #clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_impurity_decrease=0.001, random_state=0, class_weight='balanced') # better for smaller apt's
    t0 = time.time()
    clf_dt.fit(Xtr, y_train_1g)
    t1 = time.time()
    y_pred = clf_dt.predict(Xte)
    t2 = time.time()
    acc = accuracy_score(y_test_1g, y_pred)
    print(f"Data split complete. Trained in {t1-t0:.2f}s, predicted in {t2-t1:.2f}s")
    print(f"Test Accuracy: {acc*100:.2f}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_1g, y_pred, average='weighted', zero_division=0)
    generate_totals("Decision Tree", "1-gram", acc, precision, recall, f1_score)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    report_dt_1g = classification_report(y_test_1g, y_pred, zero_division=0)
    print("Classification Report:\n", report_dt_1g)
    cm = confusion_matrix(y_test_1g, y_pred, labels=np.unique(y_test_1g))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_1g)).plot(xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "DT_1g_cm.png"), dpi=150)
    plt.close()
    pd.DataFrame(classification_report(y_test_1g, y_pred, zero_division=0, output_dict=True)).to_csv(
        os.path.join(EXPORT_DIR, "DT_1g_report.csv"), index=False
    )

    # ---- 1-GRAM: KNN ----
    print("\n--- KNN (k=4) (1-gram) ---")
    clf_knn = KNeighborsClassifier(n_neighbors=KNN_K, metric='minkowski', p=2, n_jobs=-1)
    t0 = time.time()
    clf_knn.fit(Xtr, y_train_1g)
    t1 = time.time()
    y_pred = clf_knn.predict(Xte)
    t2 = time.time()
    acc = accuracy_score(y_test_1g, y_pred)
    print(f"Data split complete. Trained in {t1-t0:.2f}s, predicted in {t2-t1:.2f}s")
    print(f"Test Accuracy: {acc*100:.2f}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_1g, y_pred, average='weighted', zero_division=0)
    generate_totals("KNN", "1-gram", acc, precision, recall, f1_score)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    report_knn_1g = classification_report(y_test_1g, y_pred, zero_division=0)
    print("Classification Report:\n", report_knn_1g)
    cm = confusion_matrix(y_test_1g, y_pred, labels=np.unique(y_test_1g))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_1g)).plot(xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "KNN_1g_cm.png"), dpi=150)
    plt.close()
    pd.DataFrame(classification_report(y_test_1g, y_pred, zero_division=0, output_dict=True)).to_csv(
        os.path.join(EXPORT_DIR, "KNN_1g_report.csv"), index=False
    )

    # ---- 1-GRAM: SVM (basic RBF) ----
    print("\n--- SVM (RBF) (1-gram) ---")
    clf_svm = svm.SVC()  # default kernel='rbf'
    t0 = time.time()
    clf_svm.fit(Xtr, y_train_1g)
    t1 = time.time()
    y_pred = clf_svm.predict(Xte)
    t2 = time.time()
    acc = accuracy_score(y_test_1g, y_pred)
    print(f"Data split complete. Trained in {t1-t0:.2f}s, predicted in {t2-t1:.2f}s")
    print(f"Test Accuracy: {acc*100:.2f}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_1g, y_pred, average='weighted', zero_division=0)
    generate_totals("SVM", "1-gram", acc, precision, recall, f1_score)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    report_svm_1g = classification_report(y_test_1g, y_pred, zero_division=0)
    print("Classification Report:\n", report_svm_1g)
    cm = confusion_matrix(y_test_1g, y_pred, labels=np.unique(y_test_1g))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_1g)).plot(xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "SVM_1g_cm.png"), dpi=150)
    plt.close()
    pd.DataFrame(classification_report(y_test_1g, y_pred, zero_division=0, output_dict=True)).to_csv(
        os.path.join(EXPORT_DIR, "SVM_1g_report.csv"), index=False
    )


    # Repeat for 2-GRAM -------------------------------------------------------------------
    print("\n***Switching to 2-gram feature set***")
    Xtr = X_train_2g.fillna(0.0).astype(float)
    Xte = X_test_2g.fillna(0.0).astype(float)

    # ---- 2-GRAM: Decision Tree ----
    print("\n--- Decision Tree (2-gram) ---")
    #clf_dt = DecisionTreeClassifier(random_state=0)
    clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_impurity_decrease=0.001, random_state=0) # strong inital accuracy and fair for the small APT's
    #clf_dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, min_impurity_decrease=0.01, random_state=0) # more general and overfits
    #clf_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_impurity_decrease=0.001, random_state=0, class_weight='balanced') # better for smaller apt's
    t0 = time.time()
    clf_dt.fit(Xtr, y_train_2g)
    t1 = time.time()
    y_pred = clf_dt.predict(Xte)
    t2 = time.time()
    acc = accuracy_score(y_test_2g, y_pred)
    print(f"Data split complete. Trained in {t1-t0:.2f}s, predicted in {t2-t1:.2f}s")
    print(f"Test Accuracy: {acc*100:.2f}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_2g, y_pred, average='weighted', zero_division=0)
    generate_totals("Decision Tree", "2-gram", acc, precision, recall, f1_score)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    report_dt_2g = classification_report(y_test_2g, y_pred, zero_division=0)
    print("Classification Report:\n", report_dt_2g)
    cm = confusion_matrix(y_test_2g, y_pred, labels=np.unique(y_test_2g))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_2g)).plot(xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "DT_2g_cm.png"), dpi=150)
    plt.close()
    pd.DataFrame(classification_report(y_test_2g, y_pred, zero_division=0, output_dict=True)).to_csv(
        os.path.join(EXPORT_DIR, "DT_2g_report.csv"), index=False
    )

    # ---- 2-GRAM: KNN ----
    print("\n--- KNN (k=4) (2-gram) ---")
    clf_knn = KNeighborsClassifier(n_neighbors=KNN_K, metric='minkowski', p=2, n_jobs=-1)
    t0 = time.time()
    clf_knn.fit(Xtr, y_train_2g)
    t1 = time.time()
    y_pred = clf_knn.predict(Xte)
    t2 = time.time()
    acc = accuracy_score(y_test_2g, y_pred)
    print(f"Data split complete. Trained in {t1-t0:.2f}s, predicted in {t2-t1:.2f}s")
    print(f"Test Accuracy: {acc*100:.2f}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_2g, y_pred, average='weighted', zero_division=0)
    generate_totals("KNN", "2-gram", acc, precision, recall, f1_score)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    report_knn_2g = classification_report(y_test_2g, y_pred, zero_division=0)
    print("Classification Report:\n", report_knn_2g)
    cm = confusion_matrix(y_test_2g, y_pred, labels=np.unique(y_test_2g))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_2g)).plot(xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "KNN_2g_cm.png"), dpi=150)
    plt.close()
    pd.DataFrame(classification_report(y_test_2g, y_pred, zero_division=0, output_dict=True)).to_csv(
        os.path.join(EXPORT_DIR, "KNN_2g_report.csv"), index=False
    )

    # ---- 2-GRAM: SVM (basic RBF) ----
    print("\n--- SVM (RBF) (2-gram) ---")
    clf_svm = svm.SVC()  # default kernel='rbf'
    t0 = time.time()
    clf_svm.fit(Xtr, y_train_2g)
    t1 = time.time()
    y_pred = clf_svm.predict(Xte)
    t2 = time.time()
    acc = accuracy_score(y_test_2g, y_pred)
    print(f"Data split complete. Trained in {t1-t0:.2f}s, predicted in {t2-t1:.2f}s")
    print(f"Test Accuracy: {acc*100:.2f}%")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_2g, y_pred, average='weighted', zero_division=0)
    generate_totals("SVM", "2-gram", acc, precision, recall, f1_score)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    report_svm_2g = classification_report(y_test_2g, y_pred, zero_division=0)
    print("Classification Report where 'support' is number of APT samples during testing:\n", report_svm_2g)
    cm = confusion_matrix(y_test_2g, y_pred, labels=np.unique(y_test_2g))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test_2g)).plot(xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPORT_DIR, "SVM_2g_cm.png"), dpi=150)
    plt.close()
    pd.DataFrame(classification_report(y_test_2g, y_pred, zero_division=0, output_dict=True)).to_csv(
        os.path.join(EXPORT_DIR, "SVM_2g_report.csv"), index=False
    )

    print(f"\nAll reports and confusion matrices are saved under: {EXPORT_DIR}")

    # Summary output
    summary_df = pd.DataFrame(summary) # summary print
    print("\n=== Overall Classifier Comparison ===")
    print(summary_df)
    summary_path = os.path.join(EXPORT_DIR, "overall_classifier_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVE] Classifier comparison summary -> {summary_path}")

if __name__ == "__main__":
    main()