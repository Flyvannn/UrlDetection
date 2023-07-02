from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np

X = np.array(range(24))
X = X.reshape([8,3])
print(X)
y = np.append(np.ones(3),np.zeros(5))
print(y)

ros = RandomOverSampler(random_state=0)
st = SMOTE(random_state=0)

X_resampled, y_resampled = ros.fit_resample(X, y)
# X_st, y_st = st.fit_resample(X, y)

print(X_resampled)
print(y_resampled)

# print(X_st)
# print(y_st)