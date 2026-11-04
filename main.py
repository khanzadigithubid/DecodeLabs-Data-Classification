# =============================================================================
# DecodeLabs AI Industrial Training - Project 02
# Data Classification Using AI
# Topic: Iris Flower Classification with a Decision Tree
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Resolve paths relative to this script's location so outputs always
# land in the project folder regardless of where the script is invoked from.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# SECTION 1: LOAD AND UNDERSTAND THE DATASET
# =============================================================================

def load_and_explore_dataset():
    """Load the Iris dataset and display key information about it."""

    print("=" * 60)
    print("  DECODELABS PROJECT 02 – DATA CLASSIFICATION USING AI")
    print("=" * 60)

    # --- Load the Iris dataset from scikit-learn ---
    # The Iris dataset contains 150 samples of iris flowers.
    # Each sample has 4 features: sepal length, sepal width,
    # petal length, and petal width.
    # There are 3 classes (species): setosa, versicolor, virginica.
    iris = load_iris()

    # Convert to a Pandas DataFrame for easier exploration
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target                          # numeric label (0, 1, 2)
    df['species_name'] = df['species'].map(              # human-readable label
        {i: name for i, name in enumerate(iris.target_names)}
    )

    # --- Display the first few rows ---
    print("\n📋 FIRST 5 ROWS OF THE DATASET")
    print("-" * 60)
    print(df.head())

    # --- Dataset shape ---
    print(f"\n📐 DATASET SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Feature names ---
    print("\n🔑 FEATURES (Input Variables):")
    for i, name in enumerate(iris.feature_names, 1):
        print(f"   {i}. {name}")

    # --- Target labels ---
    print("\n🌸 TARGET LABELS (Classes to Predict):")
    for i, name in enumerate(iris.target_names, 0):
        count = (df['species'] == i).sum()
        print(f"   {i} → {name}  ({count} samples)")

    # --- Basic statistics ---
    print("\n📊 BASIC STATISTICS (Feature Summary)")
    print("-" * 60)
    print(df[iris.feature_names].describe().round(2))

    # --- Dataset info ---
    print("\n📁 DATASET INFO")
    print("-" * 60)
    df.info()

    return iris, df


# =============================================================================
# SECTION 2: SPLIT DATA INTO TRAINING AND TESTING SETS
# =============================================================================

def split_dataset(iris):
    """
    Split the dataset into 80% training and 20% testing.

    WHY DO WE SPLIT?
    ----------------
    - Training set (80%): The model LEARNS patterns from this data.
    - Testing set  (20%): We test the model on data it has NEVER seen
                          to measure how well it generalises to new inputs.
    - random_state=42  : Fixes the random seed so every run produces
                         the same split (reproducibility).
    """

    X = iris.data    # Features  – what we feed into the model
    y = iris.target  # Labels    – what the model should predict

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20 % goes to testing
        random_state=42     # reproducible results
    )

    print("\n" + "=" * 60)
    print("  SECTION 2: TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"\n  Total samples  : {len(X)}")
    print(f"  Training set   : {len(X_train)} samples  (80%)")
    print(f"  Testing set    : {len(X_test)}  samples  (20%)")
    print(f"\n  random_state=42 ensures the same split every run.")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 3: TRAIN, PREDICT, AND EVALUATE THE MODEL
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, iris):
    """
    Train a Decision Tree Classifier, make predictions, and evaluate results.

    DECISION TREE – HOW IT WORKS (beginner summary)
    -------------------------------------------------
    A Decision Tree asks a series of yes/no questions about the features
    (e.g. "Is petal length < 2.45 cm?"). At each step it picks the question
    that best separates the classes until it reaches a final prediction.
    It is easy to understand and visualise, making it a great starting point.
    """

    print("\n" + "=" * 60)
    print("  SECTION 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # --- Train the Decision Tree ---
    # max_depth=3 keeps the tree shallow and easy to visualise.
    # random_state=42 makes the training reproducible.
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print("\n✅ Decision Tree model trained successfully.")

    # --- Make predictions on the test set ---
    y_pred = model.predict(X_test)

    # --- Accuracy score ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 MODEL ACCURACY: {accuracy * 100:.2f}%")

    # --- Classification report ---
    print("\n📋 CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=iris.target_names
    ))

    # --- Show actual vs predicted results (first 20 rows) ---
    print("🔍 ACTUAL vs PREDICTED (first 20 test samples)")
    print("-" * 60)
    results_df = pd.DataFrame({
        'Actual Label'   : y_test[:20],
        'Predicted Label': y_pred[:20],
        'Actual Species' : [iris.target_names[i] for i in y_test[:20]],
        'Predicted Species': [iris.target_names[i] for i in y_pred[:20]],
        'Correct?'       : ['✅ Yes' if a == p else '❌ No'
                            for a, p in zip(y_test[:20], y_pred[:20])]
    })
    print(results_df.to_string(index=False))

    return model


# =============================================================================
# BONUS: PREDICT A NEW FLOWER SAMPLE
# =============================================================================

def predict_new_sample(model, iris):
    """Predict the species of a brand-new, unseen flower sample."""

    print("\n" + "=" * 60)
    print("  BONUS: PREDICT A NEW FLOWER SAMPLE")
    print("=" * 60)

    # A new flower with these measurements (not in the dataset)
    # [sepal length, sepal width, petal length, petal width]  (all in cm)
    new_samples = [
        [5.1, 3.5, 1.4, 0.2],   # likely setosa
        [6.7, 3.1, 4.7, 1.5],   # likely versicolor
        [6.3, 3.3, 6.0, 2.5],   # likely virginica
    ]

    print("\n  Sample Measurements → Predicted Species")
    print("-" * 60)
    for sample in new_samples:
        prediction = model.predict([sample])[0]
        species    = iris.target_names[prediction]
        proba      = model.predict_proba([sample])[0]
        confidence = proba[prediction] * 100
        print(f"  {sample}  →  {species:12s}  (confidence: {confidence:.1f}%)")


# =============================================================================
# BONUS: VISUALISE THE DECISION TREE
# =============================================================================

def visualise_tree(model, iris):
    """Save a readable visualisation of the trained Decision Tree."""

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=11,
        ax=ax
    )
    ax.set_title("Decision Tree – Iris Flower Classification", fontsize=14, pad=12)
    plt.tight_layout()
    out_path = os.path.join(_PROJECT_DIR, "decision_tree.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("\n🌳 Decision Tree diagram saved as 'decision_tree.png'")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Section 1 – Load and explore the dataset
    iris, df = load_and_explore_dataset()

    # Section 2 – Split into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(iris)

    # Section 3 – Train, predict, and evaluate
    model = train_and_evaluate(X_train, X_test, y_train, y_test, iris)

    # Bonus – Predict new flower samples
    predict_new_sample(model, iris)

    # Bonus – Visualise the tree
    visualise_tree(model, iris)

    print("\n" + "=" * 60)
    print("  PROJECT COMPLETE – All requirements satisfied.")
    print("=" * 60 + "\n")
