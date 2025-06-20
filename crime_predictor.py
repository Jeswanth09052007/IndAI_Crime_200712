# -*- coding: utf-8 -*-
"""Crime Prediction Pipeline using Year, State, and District"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import joblib

# ====================== 1. DATA LOADING ======================
def load_data():
    DATA_FILE = "cleaned_crime_data.csv"
    try:
        df = pd.read_csv(DATA_FILE, low_memory=False)
        df.columns = df.columns.str.strip()  # Strip whitespace from columns
        print(f"✅ Successfully loaded {DATA_FILE}")
        print(f"📊 Dataset shape: {df.shape}")
        print("🧾 Columns:", df.columns.tolist())
        print(df.head())
        return df
    except Exception as e:
        print(f"🔥 Error loading data: {str(e)}")
        exit()

# ====================== 2. DATA PREPROCESSING ======================
def preprocess_data(df):
    TARGET_COLUMN = "total_ipc_crimes"
    COLS_TO_DROP = ["other_ipc_cases"]

    for col in ['year', 'state/ut', 'district', TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"⚠️ '{col}' column not found in dataset.")

    drop_cols = [col for col in COLS_TO_DROP if col in df.columns]
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)
        print(f"🗑 Dropped columns: {drop_cols}")

    df.replace('Unknown', np.nan, inplace=True)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')

    missing_target = df[TARGET_COLUMN].isna().sum()
    print(f"Missing values in target '{TARGET_COLUMN}': {missing_target}")
    if missing_target > 0:
        median_val = df[TARGET_COLUMN].median()
        print(f"Filling missing '{TARGET_COLUMN}' values with median: {median_val}")
        df[TARGET_COLUMN].fillna(median_val, inplace=True)

    df.dropna(subset=['year', 'state/ut', 'district'], inplace=True)
    df['state/ut'].fillna(df['state/ut'].mode()[0], inplace=True)
    df['district'].fillna(df['district'].mode()[0], inplace=True)

    cat_cols = ['state/ut', 'district']
    print(f"🔁 Encoding categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"✅ Data shape after preprocessing: {df.shape}")
    return df

# ====================== 3. MODEL BUILDING ======================
def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# ====================== MAIN ======================
if __name__ == "__main__":
    print("\n🔍 Loading crime data...")
    raw_data = load_data()

    print("\n🧹 Preprocessing data...")
    processed_data = preprocess_data(raw_data)

    TARGET_COLUMN = "total_ipc_crimes"
    X = processed_data.drop(columns=[TARGET_COLUMN])
    y = processed_data[TARGET_COLUMN]

    # 💾 Save the input feature columns BEFORE transformations
    joblib.dump(X.columns.tolist(), 'columns.pkl')
    print("🧠 Saved input feature columns as 'columns.pkl'")

    numeric_features = ['year']
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_features])
    joblib.dump(scaler, 'crime_scaler.pkl')
    print("🔧 Numeric feature scaler saved as 'crime_scaler.pkl'")

    X_categorical = X[categorical_features].astype(np.float32).values
    X_final = np.hstack([X_numeric_scaled.astype(np.float32), X_categorical])

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y.astype(np.float32), test_size=0.2, random_state=42
    )

    print("\n🤖 Training model...")
    model = build_model((X_train.shape[1],))
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # Assuming 'X' is your final input DataFrame used to train the model
    # ✅ Create dummy_columns.csv with only column names for inference
    dummy_column_names = pd.DataFrame(X.columns)
    dummy_column_names.to_csv("dummy_columns.csv", index=False, header=False)
    print("📂 Saved input column structure to 'dummy_columns.csv'")



    print("\n📊 Evaluation Results:")
    loss, mae = model.evaluate(X_test, y_test)
    print(f"✅ Test MAE (Mean Absolute Error): {mae:.2f}")

    model.save('crime_prediction_model.h5')
    print("\n💾 Model saved to 'crime_prediction_model.h5'")

    # ============= SAMPLE PREDICTION =============
    sample_year = 2029
    sample_state = 'MAHARASTRA'
    sample_district = 'MUMBAI'

    print("\n🔮 Sample Prediction:")
    sample_data = pd.DataFrame({
        'year': [sample_year],
        'state/ut': [sample_state],
        'district': [sample_district]
    })

    # One-hot encode sample
    sample_data = pd.get_dummies(sample_data, columns=['state/ut', 'district'], drop_first=True)

    # Ensure all training columns are present
    for col in X.columns:
        if col not in sample_data.columns:
            sample_data[col] = 0
    sample_data = sample_data[X.columns]

    # Scale numeric features
    sample_data[numeric_features] = scaler.transform(sample_data[numeric_features])
    sample_np = sample_data.values.astype(np.float32)

    prediction = model.predict(sample_np)
    print(f"📈 Predicted IPC Crimes for {sample_year}, {sample_state}, {sample_district}: {prediction[0][0]:.0f}")
