import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Cargar datos
df = pd.read_csv('train.csv')

# Mostrar primeras filas
print("Primeras filas del dataset:")
print(df.head())
print("\nResumen del dataset:")
print(df.info())

# Eliminar columnas innecesarias (verifica que existan)
columns_to_drop = ['PassengerId', 'Name', 'Ticket']
if 'Cabin' in df.columns:
    columns_to_drop.append('Cabin')

df = df.drop(columns=columns_to_drop)

# Tratar valores nulos
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Codificar variables categóricas
label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Separar X e y
X = df.drop('Survived', axis=1)
y = df['Survived']

# Dividir conjunto
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_val)

print("\n--- Evaluación del modelo ---")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print("Matriz de confusión:\n", confusion_matrix(y_val, y_pred))
print("Reporte de clasificación:\n", classification_report(y_val, y_pred))
