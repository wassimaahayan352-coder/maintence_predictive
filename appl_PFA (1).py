# -- coding: utf-8 --
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np
import pytz
import matplotlib
matplotlib.use('Agg')  # KHASSA T-KON QBEL "import matplotlib.pyplot as plt"

# ============================================================
# 🟢 CONFIGURATION FLASK
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'votre_cle_secrete_ici'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///maintenance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ============================================================
# 🧱 MODELES DE LA BASE DE DONNEES
# ============================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type_input = db.Column(db.String(50))
    air_temp = db.Column(db.Float)
    proc_temp = db.Column(db.Float)
    rot_speed = db.Column(db.Float)
    torque = db.Column(db.Float)
    tool_wear = db.Column(db.Integer)
    result = db.Column(db.String(100))
    temps_restant = db.Column(db.String(100))
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

with app.app_context():
    db.create_all()

# ============================================================
# 🔮 CHARGEMENT DES MODELES ML
# ============================================================
rf_model = None
cox_model = None

RF_PATH = r"C:/Users/HP/Desktop/ia/appl_PFA/model_M_RF_1.pkl"
COX_PATH = r"C:/Users/HP/Desktop/ia/appl_PFA/coxph_1.pkl"

if os.path.exists(RF_PATH):
    try:
        with open(RF_PATH, 'rb') as f:
            rf_model = pickle.load(f)
    except Exception as e:
        print("⚠ Erreur lors du chargement du modèle RF:", e)
else:
    print(f"⚠ Modèle RF non trouvé à {RF_PATH}")

if os.path.exists(COX_PATH):
    try:
        cox_model = joblib.load(COX_PATH)
    except Exception as e:
        print("⚠ Erreur lors du chargement du modèle Cox:", e)
else:
    print(f"⚠ Modèle Cox non trouvé à {COX_PATH}")

# ============================================================
# 🔮 MAPPING TYPE MACHINE (FR)
# ============================================================
type_map = {
    0: "Élevé",
    1: "Faible",
    2: "Moyen"
}
type_inv_map = {v: k for k, v in type_map.items()}


# ============================================================
# 🔐 LOGIN & INSCRIPTION
# ============================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        action = request.form.get("action")

        if action == "register":
            if User.query.filter_by(username=username).first():
                return render_template("login.html", error="Nom d'utilisateur déjà existant ❌")
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            session['user_id'] = new_user.id
            session['username'] = new_user.username
            return redirect(url_for("index"))

        if action == "login":
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                session['user_id'] = user.id
                session['username'] = user.username
                return redirect(url_for("index"))
            else:
                return render_template("login.html", error="Nom d'utilisateur ou mot de passe incorrect ❌")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ============================================================
# 🏠 PAGE PRINCIPALE – PREDICTION
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    result = None
    temps_restant = None

    if request.method == "POST":
        if rf_model is None:
            return render_template("index.html",
                                   result="❌ Modèle RF non chargé",
                                   type_map=type_map,
                                   username=session.get('username'))

        type_input = request.form["type_input"]
        air_temp = float(request.form["air_temp"])
        proc_temp = float(request.form["proc_temp"])
        rot_speed = float(request.form["rot_speed"])
        torque = float(request.form["torque"])
        tool_wear = int(request.form["tool_wear"])

        X_input = pd.DataFrame([[
            type_inv_map.get(type_input, 0),
            air_temp,
            proc_temp,
            rot_speed,
            torque,
            tool_wear
        ]], columns=[
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        ])

        try:
            panne = rf_model.predict(X_input)[0]
        except Exception as e:
            panne = 0
            print("Erreur prédiction RF:", e)

        if panne == 1:
            result = "⚠ Risque de panne détecté !"
            if cox_model is not None:
                try:
                    cox_input = X_input.copy()
                    cox_input["RUL"] = max(0, 253 - tool_wear)
                    cox_input["event"] = 1
                    surv = cox_model.predict_survival_function(cox_input)
                    arr = np.asarray(surv)
                    if arr.ndim == 2:
                        values = arr[:, 0]
                        times = np.arange(len(values))
                        below_idx = np.where(values < 0.5)[0]
                        if len(below_idx) > 0:
                            temps_restant = str(int(times[below_idx[0]]))
                        else:
                            temps_restant = "Difficile à estimer"
                    else:
                        temps_restant = "Difficile à estimer"
                except Exception as e:
                    print("Erreur estimation Cox:", e)
                    temps_restant = "Erreur estimation RUL"
            else:
                temps_restant = "Modèle Cox non chargé"
        else:
            result = "✅ Aucun risque de panne détecté."
            temps_restant = "Non applicable"

        new_pred = Prediction(
            user_id=session['user_id'],
            type_input=type_input,
            air_temp=air_temp,
            proc_temp=proc_temp,
            rot_speed=rot_speed,
            torque=torque,
            tool_wear=tool_wear,
            result=result,
            temps_restant=str(temps_restant)
        )
        db.session.add(new_pred)
        db.session.commit()

    last_pred = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.id.desc()).first()

    return render_template("index.html",
                           result=result,
                           temps_restant=temps_restant,
                           type_map=type_map,
                           username=session.get('username'),
                           last_pred=last_pred)

# ============================================================
# 📊 ANALYSE & VISUALISATION
# ============================================================
@app.route("/analyze")
def analyze():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    timezone = pytz.timezone("Africa/Casablanca")
    date_now = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S")

    user_id = session.get("user_id")
    preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.id.desc()).all()

    if not preds:
        return render_template("analyze.html", empty=True, date=date_now, username=session.get("username"))

    # DataFrame pour historique
    df = pd.DataFrame([{
        "ID": p.id,
        "Vitesse": p.rot_speed,
        "Torque": p.torque,
        "Wear": p.tool_wear,
        "Résultat": "Panne ⚠" if "⚠" in p.result else "Non panne ✅",
        "RUL": int(p.temps_restant) if p.temps_restant and p.temps_restant.isdigit() else "N/A",
        "Date": p.date.strftime("%d/%m/%Y %H:%M")
    } for p in preds])

    total_preds = len(df)
    nb_pannes = (df["Résultat"].str.contains("Panne")).sum()
    wear_moyen = round(df["Wear"].mean(), 2)

    # Recommandation
    last_pred = preds[0]
    if "⚠" in last_pred.result:
        recommendation_text = "Action requise : vérifier la machine rapidement pour éviter la panne."
    else:
        recommendation_text = "Machine Non panne . Continuer la surveillance régulière."

    # Helper pour graphes
    def fig_to_base64(fig):
        img = io.BytesIO()
        fig.savefig(img, format="png", transparent=True, bbox_inches="tight")
        img.seek(0)
        plt.close(fig)
        return base64.b64encode(img.getvalue()).decode()

    # ==== Graphe RUL ====
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    try:
        minutes = int(last_pred.temps_restant) if last_pred.temps_restant.isdigit() else 0
    except:
        minutes = 0
    max_minutes = 253
    color = "#2ecc71" if minutes > 60 else "#e74c3c"
    ax1.pie([minutes, max(0, max_minutes - minutes)], colors=[color, "#34495e"],
            startangle=90, wedgeprops={'width': 0.3})
    plt.text(0, 0, f"{minutes} min", ha='center', va='center', fontsize=18, fontweight='bold', color='white')
    ax1.set_title("Temps Restant (RUL)", color="white", fontsize=14)
    rul_chart = fig_to_base64(fig1)

    # ==== Graphe Facteurs ====
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    facteurs = {
        "Usure": (last_pred.tool_wear / 250) * 100,
        "Couple": (last_pred.torque / 80) * 100,
        "Vitesse": (last_pred.rot_speed / 3000) * 100,
        "Temp": (last_pred.proc_temp / 315) * 100
    }
    keys = list(facteurs.keys())
    values = list(facteurs.values())
    colors = ["#3498db", "#b1631e", "#8c50a4", "#943329"]
    ax2.barh(keys, values, color=colors)
    ax2.set_xlim(0, 110)
    ax2.set_title("Facteurs Impactant la Panne (%)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    factors_chart = fig_to_base64(fig2)

    # ==== Graphe Radar ====
    fig3, ax3 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    categories = ['Temp', 'Vitesse', 'Couple', 'Usure']
    vals = [last_pred.proc_temp/315, last_pred.rot_speed/3000, last_pred.torque/80, last_pred.tool_wear/250]
    vals += vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    ax3.plot(angles, vals, color='#9b59b6', linewidth=2)
    ax3.fill(angles, vals, color='#9b59b6', alpha=0.3)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, color='white')
    ax3.set_yticklabels([])
    ax3.set_title("État Global", color="white", pad=20)
    radar_chart = fig_to_base64(fig3)


    return render_template(
        "analyze.html",
        empty=False,
        username=session.get("username"),
        date=date_now,
        total_preds=total_preds,
        nb_pannes=nb_pannes,
        wear_moyen=wear_moyen,
        recommendation_text=recommendation_text,
        rul_chart=rul_chart,
        factors_chart=factors_chart,
        radar_chart=radar_chart
    )


# ================= HISTORIQUE =================
@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    preds = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.id.desc()).all()
    date_now = datetime.now(pytz.timezone("Africa/Casablanca")).strftime("%d/%m/%Y à %H:%M")

    return render_template('history.html',
                           predictions=preds,
                           username=session['username'],
                           date=date_now)


# ============================================================
# ============================================================
# 🚀 LANCEMENT
# ============================================================

# 1. Had l-function khassha t-kon l-foq (Qbel l-run)
@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()

# 2. L-run khasso y-kon houwa l-akhir f l-file
if __name__ == '__main__':
    app.run(debug=True, threaded=True)