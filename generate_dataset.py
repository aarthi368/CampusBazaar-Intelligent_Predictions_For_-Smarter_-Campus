import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

categories = ['Books', 'Calculators', 'Lab Equipment', 'Electronics', 'Stationery', 'Bags', 'Project Materials', 'Other']
conditions = ['Like New', 'Good', 'Fair', 'Poor']
departments = ['CSE', 'ECE', 'ME', 'CE', 'EEE', 'IT', 'BIO', 'CHEM']
semesters = [1, 2, 3, 4, 5, 6, 7, 8]
posting_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
demand_levels = ['Low', 'Medium', 'High']
claim_speeds = ['Within 1 Day', 'Within 1 Week', 'Low Probability']

item_templates = {
    'Books': [
        'Data Structures Book', 'Engineering Mathematics Book', 'Programming in Python',
        'Digital Electronics', 'Operating Systems', 'Computer Networks', 'DBMS Book',
        'Algorithms Book', 'Discrete Mathematics', 'Compiler Design Book',
        'Signals and Systems', 'Control Systems', 'Machine Learning Book',
        'Engineering Physics', 'Engineering Chemistry'
    ],
    'Calculators': [
        'Scientific Calculator', 'Graphing Calculator', 'Basic Calculator',
        'Casio FX-991ES', 'HP 35s Scientific', 'Sharp Calculator'
    ],
    'Lab Equipment': [
        'Lab Coat', 'Safety Goggles', 'Vernier Caliper', 'Multimeter',
        'Breadboard', 'Soldering Iron', 'Oscilloscope Probe', 'Test Tubes Set',
        'Measuring Flask', 'Bunsen Burner'
    ],
    'Electronics': [
        'Arduino Kit', 'Raspberry Pi', 'Laptop Stand', 'Wireless Mouse',
        'USB Hub', 'HDMI Cable', 'Portable Charger', 'Earphones',
        'Keyboard', 'Webcam'
    ],
    'Stationery': [
        'Graph Sheets', 'Engineering Drawing Sheets', 'Drafting Compass Set',
        'Rulers Set', 'Technical Pens', 'Highlighters Set', 'Record Notebook',
        'Lab Manual', 'Sketch Pens', 'Geometry Box'
    ],
    'Bags': [
        'College Backpack', 'Laptop Bag', 'Drawstring Bag', 'Sling Bag',
        'Trolley Bag'
    ],
    'Project Materials': [
        'Engineering Drawing Kit', 'Mini Drill Machine', 'Soldering Kit',
        'Prototype Board', 'Sensor Module Set', 'Motor Driver Kit',
        'LED Strip', 'Jumper Wires Bundle'
    ],
    'Other': [
        'Water Bottle', 'College ID Card Holder', 'Extension Board',
        'Table Lamp', 'Sticky Notes', 'Planner Diary'
    ]
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
    else: score += 0
    
    price_min, price_max = price_ranges[category]
    price_pct = (price - price_min) / (price_max - price_min + 1)
    if price_pct < 0.3: score += 30
    elif price_pct < 0.6: score += 15
    else: score += 0
    
    if condition == 'Like New': score += 25
    elif condition == 'Good': score += 15
    elif condition == 'Fair': score += 5
    else: score += 0
    
    noise = random.randint(-10, 10)
    score += noise
    
    if score >= 60: return 'Within 1 Day'
    elif score >= 35: return 'Within 1 Week'
    else: return 'Low Probability'

records = []
for i in range(600):
    cat = random.choice(categories)
    items_list = item_templates[cat]
    item_name = random.choice(items_list)
    price_min, price_max = price_ranges[cat]
    price = random.randint(price_min, price_max)
    price = round(price / 10) * 10
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
    
    records.append({
        'id': i + 1, 'item_name': item_name, 'category': cat, 'price': price,
        'condition': condition, 'department': dept, 'semester': semester,
        'posting_day': day, 'demand_level': demand, 'claim_speed': claim_speed
    })

df = pd.DataFrame(records)
os.makedirs('model', exist_ok=True)
df.to_csv('model/dataset.csv', index=False)
print(f"Dataset created: {len(df)} records")
print(df['claim_speed'].value_counts())
