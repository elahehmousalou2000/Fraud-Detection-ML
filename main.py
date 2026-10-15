from sklearn.ensemble import RandomForestClassifier

X = [[100, 1], [200, 1], [5000, 20], [7000, 30], [300, 2]]
y = [0, 0, 1, 1, 0]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print(model.predict([[6000, 25]]))
