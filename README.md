# CampusBazar 🏪

**An Intelligent Student Marketplace with ML-Based Claim Prediction**

CampusBazar is a student-only campus marketplace where students can buy, sell, and exchange items — powered by a machine learning model that predicts how quickly each listing will be claimed.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML Model
```bash
python model/train_model.py
```
This will:
- Generate a 600-record realistic dataset (`model/dataset.csv`)
- Train 4 ML models: Logistic Regression, Decision Tree, Random Forest, KNN
- Select the best model by F1 score
- Save it as `model/model.pkl`
- Export metrics to `model/ml_metrics.json`

### 3. Run the Application
```bash
python app.py
```

Visit: **http://localhost:5000**

---

## 📁 Project Structure

```
CampusBazar/
├── app.py                  # Flask application
├── requirements.txt
├── model/
│   ├── train_model.py      # ML training script
│   ├── dataset.csv         # Auto-generated dataset
│   ├── model.pkl           # Trained model
│   └── ml_metrics.json     # Model evaluation results
├── data/
│   └── listings.csv        # Marketplace listings
├── templates/
│   ├── base.html           # Base layout + navbar
│   ├── index.html          # Home page
│   ├── marketplace.html    # Marketplace with filters
│   ├── item_details.html   # Item detail page
│   ├── list_item.html      # 4-step listing form
│   ├── prediction.html     # ML prediction result
│   ├── my_listings.html    # User's listings
│   ├── wishlist.html       # Saved items
│   ├── dashboard.html      # Analytics dashboard
│   ├── ml_insights.html    # ML explanation page
│   └── about.html          # About page
├── static/
│   ├── css/style.css       # Complete design system
│   └── js/script.js        # Global JavaScript
└── README.md
```

---

## 🎨 Design System

| Color | Hex | Usage |
|---|---|---|
| Olive | `#556B2F` | Primary actions, badges |
| Dark Olive | `#3F4F23` | Headings, hover states |
| Terracotta | `#C76B4A` | Highlights, CTAs |
| Cream | `#FAF7F0` | Page backgrounds |
| Charcoal | `#292524` | Body text |
| Warm Gray | `#78716C` | Subtext, borders |

---

## 🤖 Machine Learning

- **Dataset**: 600 synthetic campus marketplace records
- **Features**: Price, Category, Condition, Department, Semester, Posting Day, Demand Level
- **Target**: Claim Speed (Within 1 Day / Within 1 Week / Low Probability)
- **Models**: Logistic Regression, Decision Tree, Random Forest, KNN
- **Best Model**: Automatically selected by highest weighted F1 score

---

## 📱 Pages

| Route | Page |
|---|---|
| `/` | Home |
| `/marketplace` | Browse & Filter Items |
| `/item/<id>` | Item Details |
| `/list-item` | Multi-Step Listing Form |
| `/prediction` | ML Prediction Result |
| `/my-listings` | Manage Your Listings |
| `/wishlist` | Saved Items |
| `/dashboard` | Analytics Dashboard |
| `/ml-insights` | ML Model Explanation |
| `/about` | About CampusBazar |
