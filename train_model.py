import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Generate dataset inline ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

categories = ['Books', 'Calculators', 'Lab Equipment', 'Electronics', 'Stationery', 'Bags', 'Project Materials', 'Other']
conditions = ['Like New', 'Good', 'Fair', 'Poor']
departments = ['CSE', 'ECE', 'ME', 'CE', 'EEE', 'IT', 'BIO', 'CHEM']
semesters = [1, 2, 3, 4, 5, 6, 7, 8]
posting_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
demand_levels = ['Low', 'Medium', 'High']

item_templates = {
    'Books': ['Data Structures Book', 'Engineering Mathematics Book', 'Programming in Python',
              'Digital Electronics', 'Operating Systems', 'Computer Networks', 'DBMS Book',
              'Algorithms Book', 'Discrete Mathematics', 'Compiler Design Book',
              'Signals and Systems', 'Control Systems', 'Machine Learning Book',
              'Engineering Physics', 'Engineering Chemistry'],
    'Calculators': ['Scientific Calculator', 'Graphing Calculator', 'Basic Calculator',
                    'Casio FX-991ES', 'HP 35s Scientific', 'Sharp Calculator'],
    'Lab Equipment': ['Lab Coat', 'Safety Goggles', 'Vernier Caliper', 'Multimeter',
                      'Breadboard', 'Soldering Iron', 'Oscilloscope Probe', 'Test Tubes Set',
                      'Measuring Flask', 'Bunsen Burner'],
    'Electronics': ['Arduino Kit', 'Raspberry Pi', 'Laptop Stand', 'Wireless Mouse',
                    'USB Hub', 'HDMI Cable', 'Portable Charger', 'Earphones', 'Keyboard', 'Webcam'],
    'Stationery': ['Graph Sheets', 'Engineering Drawing Sheets', 'Drafting Compass Set',
                   'Rulers Set', 'Technical Pens', 'Highlighters Set', 'Record Notebook',
                   'Lab Manual', 'Sketch Pens', 'Geometry Box'],
    'Bags': ['College Backpack', 'Laptop Bag', 'Drawstring Bag', 'Sling Bag', 'Trolley Bag'],
    'Project Materials': ['Engineering Drawing Kit', 'Mini Drill Machine', 'Soldering Kit',
                          'Prototype Board', 'Sensor Module Set', 'Motor Driver Kit',
                          'LED Strip', 'Jumper Wires Bundle'],
    'Other': ['Water Bottle', 'College ID Card Holder', 'Extension Board',
              'Table Lamp', 'Sticky Notes', 'Planner Diary']
}

price_ranges = {
    'Books': (80, 600), 'Calculators': (200, 1200), 'Lab Equipment': (150, 2000),
    'Electronics': (300, 3000), 'Stationery': (30, 400), 'Bags': (250, 1500),
    'Project Materials': (200, 2500), 'Other': (50, 800)
}

category_demand_bias = {
    'Books': 'High', 'Calculators': 'High', 'Electronics': 'High',
    'Lab Equipment': 'Medium', 'Project Materials': 'Medium',
    'Stationery': 'Medium', 'Bags': 'Low', 'Other': 'Low'
}

def compute_claim_speed(price, condition, demand_level, category):
    score = 0
    if demand_level == 'High': score += 40
    elif demand_level == 'Medium': score += 20
    if category in price_ranges:
        price_min, price_max = price_ranges[category]
        price_pct = (price - price_min) / (price_max - price_min + 1)
        if price_pct < 0.3: score += 30
        elif price_pct < 0.6: score += 15
    if condition == 'Like New': score += 25
    elif condition == 'Good': score += 15
    elif condition == 'Fair': score += 5
    score += random.randint(-8, 8)
    if score >= 60: return 'Within 1 Day'
    elif score >= 35: return 'Within 1 Week'
    else: return 'Low Probability'

records = []
for i in range(600):
    cat = random.choice(categories)
    item_name = random.choice(item_templates[cat])
    price_min, price_max = price_ranges[cat]
    price = round(random.randint(price_min, price_max) / 10) * 10
    condition = random.choices(conditions, weights=[20, 45, 25, 10])[0]
    dept = random.choice(departments)
    semester = random.choice(semesters)
    day = random.choice(posting_days)
    base_demand = category_demand_bias[cat]
    if random.random() < 0.15:
        demand = random.choice(demand_levels)
    else:
        if base_demand == 'High':
            demand = random.choices(demand_levels, weights=[10, 30, 60])[0]
        elif base_demand == 'Medium':
            demand = random.choices(demand_levels, weights=[20, 50, 30])[0]
        else:
            demand = random.choices(demand_levels, weights=[50, 35, 15])[0]
    claim_speed = compute_claim_speed(price, condition, demand, cat)
    records.append({'id': i+1, 'item_name': item_name, 'category': cat, 'price': price,
                    'condition': condition, 'department': dept, 'semester': semester,
                    'posting_day': day, 'demand_level': demand, 'claim_speed': claim_speed})

df = pd.DataFrame(records)
os.makedirs('model', exist_ok=True)
df.to_csv('model/dataset.csv', index=False)
print(f"Dataset generated: {len(df)} records")
print(df['claim_speed'].value_counts())

# ── Feature Engineering ───────────────────────────────────────────────────────
le_cat = LabelEncoder()
le_cond = LabelEncoder()
le_dept = LabelEncoder()
le_day = LabelEncoder()
le_demand = LabelEncoder()

df['cat_enc'] = le_cat.fit_transform(df['category'])
df['cond_enc'] = le_cond.fit_transform(df['condition'])
df['dept_enc'] = le_dept.fit_transform(df['department'])
df['day_enc'] = le_day.fit_transform(df['posting_day'])
df['demand_enc'] = le_demand.fit_transform(df['demand_level'])

feature_cols = ['price', 'cat_enc', 'cond_enc', 'dept_enc', 'semester', 'day_enc', 'demand_enc']
X = df[feature_cols].values
y = df['claim_speed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Train Models ──────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

metrics_results = {}
best_model_name = None
best_f1 = -1
best_model_obj = None

print("\n--- Model Evaluation ---")
for name, model in models.items():
    if name in ['Logistic Regression', 'KNN']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics_results[name] = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4)
    }
    print(f"{name}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

print(f"\nBest Model: {best_model_name} (F1={best_f1:.4f})")

# ── Feature Importance ────────────────────────────────────────────────────────
feature_importance = {}
if hasattr(best_model_obj, 'feature_importances_'):
    importances = best_model_obj.feature_importances_
    feature_names = ['Price', 'Category', 'Condition', 'Department', 'Semester', 'Posting Day', 'Demand Level']
    feature_importance = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}

# ── Class Distribution ─────────────────────────────────────────────────────────
class_dist = df['claim_speed'].value_counts().to_dict()

# ── Save Everything ───────────────────────────────────────────────────────────
model_data = {
    'model': best_model_obj,
    'scaler': scaler,
    'le_cat': le_cat,
    'le_cond': le_cond,
    'le_dept': le_dept,
    'le_day': le_day,
    'le_demand': le_demand,
    'feature_cols': feature_cols,
    'best_model_name': best_model_name,
    'uses_scaling': best_model_name in ['Logistic Regression', 'KNN']
}
joblib.dump(model_data, 'model/model.pkl')
print("Model saved to model/model.pkl")

ml_metrics = {
    'models': metrics_results,
    'best_model': best_model_name,
    'best_f1': round(best_f1, 4),
    'feature_importance': feature_importance,
    'class_distribution': class_dist,
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'feature_names': ['Price', 'Category', 'Condition', 'Department', 'Semester', 'Posting Day', 'Demand Level']
}

with open('model/ml_metrics.json', 'w') as f:
    json.dump(ml_metrics, f, indent=2)
print("Metrics saved to model/ml_metrics.json")
print("\nTraining complete!")
