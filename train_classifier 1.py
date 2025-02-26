import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.9, shuffle=True, stratify=labels)
model = RandomForestClassifier(warm_start=True, oob_score=True, n_estimators=100)
# Train the model iteratively to track the out-of-bag error
errors = []
for i in range(1, 101):
    model.n_estimators = i
    model.fit(x_train, y_train)
    errors.append(1 - model.oob_score_)

# Plot the out-of-bag error over iterations
plt.plot(range(1, 101), errors)
plt.xlabel('Number of Trees')
plt.ylabel('Out-of-Bag Error')
plt.title('Out-of-Bag Error vs. Number of Trees')
plt.show()

# Save the final trained model
with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
