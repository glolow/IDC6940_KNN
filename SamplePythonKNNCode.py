from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (Crucial for distance-based algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Initialize and train the KNN Classifier (using K=5)
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) 
knn.fit(X_train, y_train)

# 5. Make predictions
y_pred = knn.predict(X_test)

# 6. Evaluate accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
