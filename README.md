# 🏠 House Price Prediction System

A beginner-friendly Machine Learning web application that predicts California house prices using a **Random Forest Regression** model. Built with Python and deployed using Streamlit.

---

## 🔗 Live Demo

👉 **[Click here to try the app](https://house-price-pridiction-app-26.streamlit.app/)**

> Replace the link above with your actual Streamlit deployment URL.

---

## 📌 About the Project

This project is an end-to-end supervised machine learning application built as part of learning regression concepts. The app takes house-related input features and predicts the estimated house price in real time.

The model is trained on the **California Housing Dataset** (built into scikit-learn) which contains data from the 1990 California census.

---

## 🧠 How It Works

1. The app loads the California Housing dataset directly from scikit-learn
2. Data is preprocessed using `StandardScaler`
3. A `RandomForestRegressor` model is trained on 80% of the data
4. The trained model is cached using `@st.cache_resource` so it only trains once
5. User inputs values via sliders → model predicts the house price instantly

---

## 🗂️ Project Structure

```
house-price-predictor/
├── app.py                  # Main Streamlit web application
├── california_dataset.py   # Dataset loading and preprocessing (if separate)
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from GitHub
└── README.md               # Project documentation (this file)
```

---

## 📊 Dataset — California Housing

| Feature | Description |
|---|---|
| `MedInc` | Median income of the block group |
| `HouseAge` | Median age of houses in the block |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| **Price** | **Median house value (in $100,000s) — this is what we predict** |

- Total records: **20,640**
- No missing values
- Target variable: house price in units of $100,000

---

## 🤖 Model Details

| Detail | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Number of trees | 100 |
| Train / Test split | 80% / 20% |
| Scaling | StandardScaler |
| Evaluation metrics | MAE, RMSE, R² Score |

**Why Random Forest?**
Instead of one decision tree, Random Forest builds 100 trees — each trained on a slightly different random subset of the data. The final prediction is the average of all trees. This reduces overfitting and gives significantly better accuracy than a single Linear Regression model.

| Model | R² Score (approx.) |
|---|---|
| Linear Regression | ~0.60 |
| Random Forest | ~0.80+ |

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/house-price-predictor.git
cd house-price-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
scikit-learn
numpy
pandas
joblib
```

Install all at once with:
```bash
pip install -r requirements.txt
```

---

## 📸 App Preview

> You can add a screenshot here after deployment.
> In GitHub: Edit this README → click the image icon → upload your screenshot.

---

## 🛠️ Tech Stack

- **Python** — core programming language
- **scikit-learn** — machine learning model and preprocessing
- **pandas** — data manipulation
- **numpy** — numerical operations
- **Streamlit** — web app framework
- **Streamlit Community Cloud** — free deployment platform

---

## 📈 What I Learned

- End-to-end supervised machine learning workflow
- Difference between Linear Regression and Random Forest
- Data preprocessing with StandardScaler
- Splitting data into train/test sets
- Evaluating regression models using MAE, RMSE, and R²
- Building and deploying a web app with Streamlit

---

## 👤 Author

**Jainil Chavda**
- GitHub: [@your-username](https://github.com/Jainil26)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/jainil-chavda/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
