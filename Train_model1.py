# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from joblib import dump

# Load your dataset
# Assuming your dataset has columns like 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism', and 'Personality'
df = pd.read_csv(r'C:\Users\vinee\OneDrive\Desktop\New pp\personality_dataset.csv')

# Split the data into features (X) and target (y)
x = df.drop('Personality', axis=1)
y = df['Personality']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

dump(model, 'personality_model1.joblib')