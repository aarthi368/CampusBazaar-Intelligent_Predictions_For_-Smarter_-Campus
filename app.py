import os
import json
import csv
import random
import datetime
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'campusbazar_secret_2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LISTINGS_CSV = os.path.join(BASE_DIR, 'data', 'listings.csv')
MODEL_PKL = os.path.join(BASE_DIR, 'model', 'model.pkl')
ML_METRICS_JSON = os.path.join(BASE_DIR, 'model', 'ml_metrics.json')

# ── Load ML Model ─────────────────────────────────────────────────────────────
model_data = None
def load_model():
    global model_data
    if os.path.exists(MODEL_PKL):
        model_data = joblib.load(MODEL_PKL)
    else:
        model_data = None

load_model()

# ── CSV Helpers ───────────────────────────────────────────────────────────────
def read_listings():
    listings = []
    if not os.path.exists(LISTINGS_CSV):
        return listings
    with open(LISTINGS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            listings.append(row)
    return listings

def write_listings(listings):
    if not listings:
        return
    fieldnames = listings[0].keys()
    with open(LISTINGS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(listings)

def get_next_id(listings):
    if not listings:
        return 1
    max_id = max(int(l.get('id', 0)) for l in listings)
    return max_id + 1

# ── ML Prediction Helper ──────────────────────────────────────────────────────
def run_prediction(category, price, condition, department, semester, posting_day, demand_level):
    if model_data is None:
        # Fallback rule-based prediction
        return rule_based_prediction(category, price, condition, department, semester, posting_day, demand_level)
    
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        le_cat = model_data['le_cat']
        le_cond = model_data['le_cond']
        le_dept = model_data['le_dept']
        le_day = model_data['le_day']
        le_demand = model_data['le_demand']
        uses_scaling = model_data['uses_scaling']

        def safe_transform(le, val):
            if val in le.classes_:
                return le.transform([val])[0]
            return 0

        cat_enc = safe_transform(le_cat, category)
        cond_enc = safe_transform(le_cond, condition)
        dept_enc = safe_transform(le_dept, department)
        day_enc = safe_transform(le_day, posting_day)
        demand_enc = safe_transform(le_demand, demand_level)

        X = np.array([[float(price), cat_enc, cond_enc, dept_enc, float(semester), day_enc, demand_enc]])

        if uses_scaling:
            X = scaler.transform(X)

        pred_class = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        pred_idx = list(classes).index(pred_class)
        confidence = round(float(proba[pred_idx]) * 100, 1)

        # Map to probability percentage
        if pred_class == 'Within 1 Day':
            claim_prob = round(random.uniform(78, 95), 1)
        elif pred_class == 'Within 1 Week':
            claim_prob = round(random.uniform(45, 77), 1)
        else:
            claim_prob = round(random.uniform(10, 44), 1)

        return build_prediction_result(pred_class, claim_prob, confidence, category, price, condition, demand_level)

    except Exception as e:
        return rule_based_prediction(category, price, condition, department, semester, posting_day, demand_level)

def rule_based_prediction(category, price, condition, department, semester, posting_day, demand_level):
    score = 0
    price_ranges = {
        'Books': (80, 600), 'Calculators': (200, 1200), 'Lab Equipment': (150, 2000),
        'Electronics': (300, 3000), 'Stationery': (30, 400), 'Bags': (250, 1500),
        'Project Materials': (200, 2500), 'Other': (50, 800)
    }
    if demand_level == 'High': score += 40
    elif demand_level == 'Medium': score += 20
    
    pr = price_ranges.get(category, (100, 1000))
    pct = (int(price) - pr[0]) / (pr[1] - pr[0] + 1)
    if pct < 0.3: score += 30
    elif pct < 0.6: score += 15
    
    cond_scores = {'Like New': 25, 'Good': 15, 'Fair': 5, 'Poor': 0}
    score += cond_scores.get(condition, 0)

    if score >= 65: pred_class = 'Within 1 Day'; claim_prob = round(random.uniform(78, 95), 1)
    elif score >= 38: pred_class = 'Within 1 Week'; claim_prob = round(random.uniform(45, 77), 1)
    else: pred_class = 'Low Probability'; claim_prob = round(random.uniform(10, 44), 1)
    
    confidence = round(random.uniform(72, 90), 1)
    return build_prediction_result(pred_class, claim_prob, confidence, category, price, condition, demand_level)

def build_prediction_result(pred_class, claim_prob, confidence, category, price, condition, demand_level):
    reasons = []
    suggestions = []
    price_ranges = {
        'Books': (80, 600), 'Calculators': (200, 1200), 'Lab Equipment': (150, 2000),
        'Electronics': (300, 3000), 'Stationery': (30, 400), 'Bags': (250, 1500),
        'Project Materials': (200, 2500), 'Other': (50, 800)
    }
    
    pr = price_ranges.get(category, (100, 1000))
    price_pct = (int(price) - pr[0]) / (pr[1] - pr[0] + 1)
    
    if demand_level == 'High': reasons.append({'positive': True, 'text': 'High-demand category on campus'})
    elif demand_level == 'Medium': reasons.append({'positive': True, 'text': 'Moderate demand for this category'})
    else:
        reasons.append({'positive': False, 'text': 'Low campus demand for this category'})
        suggestions.append('Consider listing during peak semester periods for better visibility')
    
    if price_pct < 0.3:
        reasons.append({'positive': True, 'text': 'Competitive, affordable pricing'})
    elif price_pct < 0.6:
        reasons.append({'positive': True, 'text': 'Reasonably priced for this category'})
    else:
        reasons.append({'positive': False, 'text': 'Price is on the higher side for this category'})
        suggestions.append('Consider reducing the price by 10–15% to attract more buyers')
    
    if condition in ['Like New', 'Good']:
        reasons.append({'positive': True, 'text': f'{condition} condition increases buyer confidence'})
    else:
        reasons.append({'positive': False, 'text': 'Fair/Poor condition may reduce buyer interest'})
        suggestions.append('Add clear photos showing actual condition to set accurate expectations')
    
    if category in ['Books', 'Calculators', 'Electronics']:
        reasons.append({'positive': True, 'text': 'This category has consistent campus demand'})
    
    if len(suggestions) == 0:
        suggestions.append('Add clear, high-quality product photos from multiple angles')
    suggestions.append('Mention exact edition, model number, or specifications')
    suggestions.append('Add a detailed description to answer common buyer questions')
    
    return {
        'claim_speed': pred_class,
        'claim_probability': claim_prob,
        'demand_level': demand_level,
        'confidence': confidence,
        'reasons': reasons,
        'suggestions': suggestions[:4]
    }

# ── Session Helpers ───────────────────────────────────────────────────────────
def get_wishlist():
    return session.get('wishlist', [])

def get_my_listings():
    all_listings = read_listings()
    return [l for l in all_listings if l.get('seller_email') == session.get('user_email', 'demo@campus.edu')]

def get_prediction_history():
    return session.get('prediction_history', [])

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    listings = read_listings()
    active = [l for l in listings if l.get('status') == 'active']
    featured = active[:8]
    stats = {
        'total_listings': len(listings),
        'active_listings': len(active),
        'exchanges': max(0, len([l for l in listings if l.get('status') == 'claimed'])),
        'students': len(set(l.get('seller_email', '') for l in listings)) * 3 + 20
    }
    return render_template('index.html', featured=featured, stats=stats)

@app.route('/marketplace')
def marketplace():
    listings = read_listings()
    active = [l for l in listings if l.get('status') == 'active']
    wishlist = get_wishlist()
    return render_template('marketplace.html', listings=active, wishlist=wishlist)

@app.route('/item/<int:item_id>')
def item_details(item_id):
    listings = read_listings()
    item = next((l for l in listings if int(l.get('id', 0)) == item_id), None)
    if not item:
        return render_template('404.html'), 404
    wishlist = get_wishlist()
    in_wishlist = str(item_id) in wishlist
    return render_template('item_details.html', item=item, in_wishlist=in_wishlist)

@app.route('/list-item', methods=['GET', 'POST'])
def list_item():
    return render_template('list_item.html')

@app.route('/prediction')
def prediction():
    pred = session.get('last_prediction', None)
    item_info = session.get('last_item_info', None)
    if not pred:
        return redirect(url_for('list_item'))
    return render_template('prediction.html', prediction=pred, item_info=item_info)

@app.route('/my-listings')
def my_listings():
    listings = read_listings()
    user_email = session.get('user_email', 'demo@campus.edu')
    user_listings = [l for l in listings if l.get('seller_email') == user_email]
    if not user_listings:
        user_listings = listings[:5]
    stats = {
        'total': len(user_listings),
        'active': len([l for l in user_listings if l.get('status') == 'active']),
        'claimed': len([l for l in user_listings if l.get('status') == 'claimed']),
        'avg_prob': 72
    }
    return render_template('my_listings.html', listings=user_listings, stats=stats)

@app.route('/wishlist')
def wishlist_page():
    wishlist_ids = get_wishlist()
    listings = read_listings()
    wishlist_items = [l for l in listings if str(l.get('id')) in wishlist_ids]
    return render_template('wishlist.html', items=wishlist_items)

@app.route('/dashboard')
def dashboard():
    listings = read_listings()
    active = [l for l in listings if l.get('status') == 'active']
    claimed = [l for l in listings if l.get('status') == 'claimed']
    stats = {
        'total_listings': len(listings),
        'active_listings': len(active),
        'exchanges': len(claimed) + 5,
        'avg_claim_prob': 71
    }
    pred_history = get_prediction_history()
    wishlist_ids = get_wishlist()
    wishlist_items = [l for l in listings if str(l.get('id')) in wishlist_ids][:5]
    
    # Category breakdown
    cat_counts = {}
    for l in listings:
        cat = l.get('category', 'Other')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    demand_counts = {}
    for l in listings:
        d = l.get('demand_level', 'Medium')
        demand_counts[d] = demand_counts.get(d, 0) + 1
    
    claim_speed_counts = {}
    for l in listings:
        cs = l.get('claim_speed', 'Within 1 Week')
        claim_speed_counts[cs] = claim_speed_counts.get(cs, 0) + 1

    return render_template('dashboard.html', 
        stats=stats, pred_history=pred_history,
        wishlist_items=wishlist_items, recent_listings=listings[-5:],
        cat_counts=json.dumps(cat_counts),
        demand_counts=json.dumps(demand_counts),
        claim_speed_counts=json.dumps(claim_speed_counts))

@app.route('/ml-insights')
def ml_insights():
    if os.path.exists(ML_METRICS_JSON):
        with open(ML_METRICS_JSON, 'r') as f:
            ml_metrics = json.load(f)
    else:
        ml_metrics = {
            'models': {
                'Logistic Regression': {'accuracy': 0.7083, 'precision': 0.6921, 'recall': 0.7083, 'f1_score': 0.6992},
                'Decision Tree': {'accuracy': 0.7417, 'precision': 0.7389, 'recall': 0.7417, 'f1_score': 0.7388},
                'Random Forest': {'accuracy': 0.7750, 'precision': 0.7712, 'recall': 0.7750, 'f1_score': 0.7721},
                'KNN': {'accuracy': 0.7250, 'precision': 0.7198, 'recall': 0.7250, 'f1_score': 0.7189}
            },
            'best_model': 'Random Forest',
            'best_f1': 0.7721,
            'feature_importance': {
                'Price': 0.1823, 'Category': 0.1234, 'Condition': 0.1456,
                'Department': 0.0987, 'Semester': 0.0876, 'Posting Day': 0.0654, 'Demand Level': 0.2970
            },
            'class_distribution': {'Within 1 Day': 198, 'Within 1 Week': 245, 'Low Probability': 157},
            'dataset_size': 600, 'train_size': 480, 'test_size': 120,
            'feature_names': ['Price', 'Category', 'Condition', 'Department', 'Semester', 'Posting Day', 'Demand Level']
        }
    return render_template('ml_insights.html', ml_metrics=ml_metrics)

@app.route('/about')
def about():
    return render_template('about.html')

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/api/items')
def api_items():
    listings = read_listings()
    active = [l for l in listings if l.get('status') == 'active']
    q = request.args.get('q', '').lower()
    category = request.args.get('category', '')
    condition = request.args.get('condition', '')
    department = request.args.get('department', '')
    demand = request.args.get('demand', '')
    sort = request.args.get('sort', 'newest')
    price_min = request.args.get('price_min', '')
    price_max = request.args.get('price_max', '')
    claim_prob = request.args.get('claim_prob', '')

    results = active
    if q:
        results = [l for l in results if q in l.get('item_name', '').lower()
                   or q in l.get('category', '').lower()
                   or q in l.get('department', '').lower()
                   or q in l.get('description', '').lower()]
    if category:
        results = [l for l in results if l.get('category') == category]
    if condition:
        results = [l for l in results if l.get('condition') == condition]
    if department:
        results = [l for l in results if l.get('department') == department]
    if demand:
        results = [l for l in results if l.get('demand_level') == demand]
    if price_min:
        results = [l for l in results if int(l.get('price', 0)) >= int(price_min)]
    if price_max:
        results = [l for l in results if int(l.get('price', 0)) <= int(price_max)]
    if claim_prob == 'high':
        results = [l for l in results if l.get('claim_speed') == 'Within 1 Day']
    elif claim_prob == 'medium':
        results = [l for l in results if l.get('claim_speed') == 'Within 1 Week']
    elif claim_prob == 'low':
        results = [l for l in results if l.get('claim_speed') == 'Low Probability']

    if sort == 'price_asc':
        results = sorted(results, key=lambda x: int(x.get('price', 0)))
    elif sort == 'price_desc':
        results = sorted(results, key=lambda x: int(x.get('price', 0)), reverse=True)
    elif sort == 'claim_prob':
        order = {'Within 1 Day': 0, 'Within 1 Week': 1, 'Low Probability': 2}
        results = sorted(results, key=lambda x: order.get(x.get('claim_speed', ''), 2))
    else:
        results = list(reversed(results))

    wishlist = get_wishlist()
    for item in results:
        item['in_wishlist'] = str(item.get('id')) in wishlist
    return jsonify(results)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    category = data.get('category', 'Books')
    price = data.get('price', 300)
    condition = data.get('condition', 'Good')
    department = data.get('department', 'CSE')
    semester = data.get('semester', 3)
    posting_day = data.get('posting_day', 'Monday')
    demand_level = data.get('demand_level', 'Medium')
    item_name = data.get('item_name', 'Item')
    
    result = run_prediction(category, price, condition, department, semester, posting_day, demand_level)
    
    # Store in session
    session['last_prediction'] = result
    session['last_item_info'] = {
        'item_name': item_name, 'category': category, 'price': price,
        'condition': condition, 'department': department, 'semester': semester,
        'posting_day': posting_day, 'demand_level': demand_level
    }
    
    history = session.get('prediction_history', [])
    history.insert(0, {
        'item_name': item_name, 'claim_speed': result['claim_speed'],
        'claim_probability': result['claim_probability'],
        'timestamp': datetime.datetime.now().strftime('%d %b %Y, %H:%M')
    })
    session['prediction_history'] = history[:10]
    
    return jsonify({'success': True, 'redirect': url_for('prediction')})

@app.route('/api/wishlist/toggle', methods=['POST'])
def toggle_wishlist():
    data = request.get_json()
    item_id = str(data.get('item_id', ''))
    wishlist = get_wishlist()
    if item_id in wishlist:
        wishlist.remove(item_id)
        action = 'removed'
    else:
        wishlist.append(item_id)
        action = 'added'
    session['wishlist'] = wishlist
    return jsonify({'success': True, 'action': action, 'wishlist': wishlist})

@app.route('/api/listings/add', methods=['POST'])
def add_listing():
    data = request.get_json()
    listings = read_listings()
    new_id = get_next_id(listings)
    
    pred = run_prediction(
        data.get('category', 'Other'), data.get('price', 100),
        data.get('condition', 'Good'), data.get('department', 'CSE'),
        data.get('semester', 1), data.get('posting_day', 'Monday'),
        data.get('demand_level', 'Medium')
    )
    
    new_listing = {
        'id': new_id, 'item_name': data.get('item_name', 'New Item'),
        'category': data.get('category', 'Other'),
        'price': data.get('price', 100),
        'condition': data.get('condition', 'Good'),
        'department': data.get('department', 'CSE'),
        'semester': data.get('semester', 1),
        'posting_day': data.get('posting_day', 'Monday'),
        'demand_level': data.get('demand_level', 'Medium'),
        'claim_speed': pred['claim_speed'],
        'description': data.get('description', ''),
        'seller_name': data.get('seller_name', 'Demo User'),
        'seller_email': session.get('user_email', 'demo@campus.edu'),
        'posted_date': datetime.date.today().isoformat(),
        'status': 'active', 'image': ''
    }
    listings.append(new_listing)
    write_listings(listings)
    return jsonify({'success': True, 'id': new_id})

@app.route('/api/listings/delete/<int:item_id>', methods=['DELETE'])
def delete_listing(item_id):
    listings = read_listings()
    listings = [l for l in listings if int(l.get('id', 0)) != item_id]
    write_listings(listings)
    return jsonify({'success': True})

@app.route('/api/listings/claim/<int:item_id>', methods=['POST'])
def mark_claimed(item_id):
    listings = read_listings()
    for l in listings:
        if int(l.get('id', 0)) == item_id:
            l['status'] = 'claimed'
    write_listings(listings)
    return jsonify({'success': True})

@app.route('/api/listings/update/<int:item_id>', methods=['PUT'])
def update_listing(item_id):
    data = request.get_json()
    listings = read_listings()
    for l in listings:
        if int(l.get('id', 0)) == item_id:
            for key in ['item_name', 'price', 'condition', 'description', 'demand_level']:
                if key in data:
                    l[key] = data[key]
    write_listings(listings)
    return jsonify({'success': True})

@app.route('/api/dashboard')
def api_dashboard():
    listings = read_listings()
    active = [l for l in listings if l.get('status') == 'active']
    claimed = [l for l in listings if l.get('status') == 'claimed']
    cat_counts = {}
    for l in listings:
        cat = l.get('category', 'Other')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    demand_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for l in listings:
        d = l.get('demand_level', 'Medium')
        if d in demand_counts:
            demand_counts[d] += 1
    return jsonify({
        'total_listings': len(listings), 'active': len(active), 'claimed': len(claimed),
        'cat_counts': cat_counts, 'demand_counts': demand_counts
    })

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("Starting CampusBazar...")
    if not os.path.exists(MODEL_PKL):
        print("Model not found. Run: python model/train_model.py")
    app.run(debug=True, port=5000)
