# IndAI_Crime_200712

# 🧠 Crime Prediction System using Deep Learning 🚨

## 📌 Project Overview

This project focuses on building an AI-based crime prediction model using **Deep Learning** techniques on **real-world Indian crime data**. The objective is to analyze historical crime data, discover patterns, and forecast future crime trends to aid in public safety planning and policy-making.

🔗 Live Demo (Optional): *[Add link if hosted]*  
📂 Dataset Source: Self-collected & merged from multiple government crime records  
📁 GitHub Repository: [You’re here!]

---

## 📊 Dataset Insights

- 📁 **1000+ CSV files** merged and cleaned
- 📈 **150,000+ rows**, **11,000+ columns**
- Features include:
  - `State`, `District`, `Crime Head`
  - `Year`, `Reported`, `Convicted`, `Crime Type`, etc.
- Extensive preprocessing for:
  - Null/missing values
  - Categorical to numerical transformation
  - Normalization

---

## 🧠 Model Architecture

- Type: **Multi-Layer Perceptron (MLP)**
- Framework: **TensorFlow / Keras**
- Layers:
  - Input Layer (Shape based on features)
  - Hidden Layers (Dense + ReLU)
  - Output Layer (Regression output - total crimes)
- Loss Function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- Activation: **ReLU**
- Epochs: Tuned between 100–300 depending on dataset split

---

## 🛠️ Tech Stack

| Category        | Tools/Tech Used            |
|----------------|-----------------------------|
| Language        | Python                     |
| Libraries       | TensorFlow, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| Visualization   | Matplotlib, Seaborn        |
| Notebook        | Jupyter Notebook            |

---

## 📈 Results & Performance

- Accurate prediction trends of IPC crime rates across years
- Handled large volume data with real-time model feedback
- Learned critical insights on over/under-reporting across districts

---

## 🔍 Key Takeaways

- ✅ Mastered end-to-end Deep Learning workflow: Data loading → Preprocessing → Model training → Evaluation
- ✅ Managed massive real-world data
- ✅ Learned importance of feature selection, normalization, and proper architecture tuning

---

## 📂 Folder Structure

crime-prediction/
│
├── data/ # Raw & cleaned datasets
├── models/ # Model training & saved weights
├── notebooks/ # Jupyter notebooks for EDA and model building
├── visuals/ # Output plots, graphs
├── README.md # Project README (You are here)
└── crime_predictor.py # Core script for model training/inference



---

## 🚀 Future Work

- Incorporate additional features like population density, poverty index, literacy rates
- Use advanced models like LSTM or Time Series forecasting
- Build a dashboard for real-time interactive visualization

---

## 🤝 Connect With Me

Made with ❤️ by **Jeswanth Reddy**  
📧 [jeswanthreddyreddem@gmail.com]  
🌐 [https://www.linkedin.com/in/jeswanth-reddy-reddem-81ba52320?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app]  
💼 Open to internships, research, and collaboration!

---


