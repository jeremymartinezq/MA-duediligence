import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import xgboost as xgb
import plotly.express as px
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Preprocessing Functions
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    # Feature engineering example
    df['new_feature'] = df['feature1'] * df['feature2']
    return df


# Financial Due Diligence using XGBoost
def financial_dd(data):
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=4)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    predictions = model.predict(X_test)

    # Visualize feature importance
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance in Financial DD")
    fig.show()

    return score


# Tax Due Diligence using RandomForest with Hyperparameter Tuning
def tax_dd(data):
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print(classification_report(y_test, y_pred))
    score = accuracy_score(y_test, y_pred)

    return score


# Compliance Due Diligence using SVM with RBF Kernel
def compliance_dd(data):
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    svm = SVC(kernel='rbf', C=1, gamma='auto')

    pipeline = Pipeline([('scaler', scaler), ('svm', svm)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(classification_report(y_test, y_pred))
    score = accuracy_score(y_test, y_pred)

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(cm, labels=dict(x="Predicted", y="True"), text_auto=True, title="Confusion Matrix - Compliance DD")
    fig.show()

    return score


# Tech Due Diligence using Deep Learning
def tech_dd(data):
    X = data.drop('target_column', axis=1)
    y = pd.get_dummies(data['target_column']).values  # One-hot encoding for classification

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

    # Plot model accuracy
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy'))
    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy'))
    fig.update_layout(title='Model Accuracy - Tech DD', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.show()

    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    return accuracy


# Cultural Due Diligence using PCA and KMeans
def cultural_dd(data):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    model = KMeans(n_clusters=3)
    clusters = model.fit_predict(data_pca)

    # Visualization of clusters
    fig = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], color=clusters, title="Cultural DD - PCA & Clustering")
    fig.show()

    return clusters


# Legal Due Diligence using RandomForest and Feature Importance
def legal_dd(data):
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    # Feature importance visualization
    importance = rf.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    importance_df.sort_values(by='Importance', ascending=False, inplace=True)

    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance - Legal DD")
    fig.show()

    return score


# Visualization
def plot_progress(completion):
    stages = ['Financial DD', 'Tax DD', 'Compliance DD', 'Tech DD', 'Cultural DD', 'Legal DD']

    fig = px.bar(x=completion, y=stages, orientation='h', text_auto=True,
                 title="M&A Due Diligence Progress", labels={'x': 'Completion Percentage'})
    fig.show()


def generate_completion_report(completion):
    plot_progress(completion)


# Main Function to Run Due Diligence
def run_due_diligence():
    print("Starting Financial DD...")
    financial_data = preprocess_data('data/financial_data.csv')
    financial_score = financial_dd(financial_data)

    print("Starting Tax DD...")
    tax_data = preprocess_data('data/tax_data.csv')
    tax_score = tax_dd(tax_data)

    print("Starting Compliance DD...")
    compliance_data = preprocess_data('data/compliance_data.csv')
    compliance_score = compliance_dd(compliance_data)

    print("Starting Tech DD...")
    tech_data = preprocess_data('data/tech_data.csv')
    tech_score = tech_dd(tech_data)

    print("Starting Cultural DD...")
    cultural_data = preprocess_data('data/cultural_data.csv')
    cultural_clusters = cultural_dd(cultural_data)

    print("Starting Legal DD...")
    legal_data = preprocess_data('data/legal_data.csv')
    legal_score = legal_dd(legal_data)

    # Example completion percentages (this should be dynamically calculated based on your needs)
    completion = [financial_score * 100, tax_score * 100, compliance_score * 100, tech_score * 100, 90,
                  legal_score * 100]

    print("Generating Completion Report...")
    generate_completion_report(completion)


if __name__ == "__main__":
    run_due_diligence()
