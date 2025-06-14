from flask import Flask, render_template, request, redirect, url_for, send_file
import sqlite3
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import io
import pandas as pd

app = Flask(__name__)
DB_PATH = 'ttf.db'
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

MTTF_THRESHOLDS = {
    "Motor": 700,
    "Hoist": 1000,
    "Dragging": 800,
    "Rigging": 1500,
    "Buckets": 1200,
    "Movement": 1000,
}

def initialize_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ttf_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine TEXT,
                component TEXT,
                value REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hmr_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine TEXT,
                component TEXT,
                hmr_value REAL,
                failure_description TEXT,
                working_hours REAL,
                maintenance_hours REAL,
                breakdown_hours REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

initialize_db()
@app.route('/')
def home():
    return redirect(url_for('welcome'))

# Welcome page to choose machine
@app.route('/welcome', methods=['GET', 'POST'])
def welcome():
    machines = get_all_machines()
    if request.method == 'POST':
        selected_machine = request.form['machine']
        return redirect(url_for('dashboard', machine=selected_machine))
    return render_template('welcome.html', machines=machines)

# Get all machine names
def get_all_machines():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT machine FROM hmr_data")
    machines = [row[0] for row in cursor.fetchall()]
    conn.close()
    return machines or ["W2000 Ajay", "W2000 Abhimanyu", "W2000 Balram"]

def get_components(machine):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT component FROM hmr_data WHERE machine = ?", (machine,))
    components = [row[0] for row in cursor.fetchall()]
    conn.close()
    return components or ["Motor"]

def get_ttf_values(machine, component):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM ttf_data WHERE machine = ? AND component = ?", (machine, component))
    ttf = [row[0] for row in cursor.fetchall()]
    conn.close()
    return np.array(ttf)

def insert_hmr_value(machine, component, hmr_value, failure_desc, working, maintenance, breakdown):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO hmr_data (machine, component, hmr_value, failure_description, working_hours, maintenance_hours, breakdown_hours)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (machine, component, hmr_value, failure_desc, working, maintenance, breakdown))

    cursor.execute("""
        SELECT hmr_value FROM hmr_data
        WHERE machine = ? AND component = ?
        ORDER BY timestamp DESC LIMIT 2
    """, (machine, component))
    hmr_vals = [row[0] for row in cursor.fetchall()]
    if len(hmr_vals) == 2:
        ttf = abs(hmr_vals[0] - hmr_vals[1])
        cursor.execute("INSERT INTO ttf_data (machine, component, value) VALUES (?, ?, ?)", (machine, component, ttf))
    conn.commit()
    conn.close()

def fetch_latest_hmr(machine, component):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT SUM(working_hours) AS working_hours,
               SUM(maintenance_hours) AS maintenance_hours,
               SUM(breakdown_hours) AS breakdown_hours
        FROM hmr_data
        WHERE machine = ? AND component = ?
    """, conn, params=(machine, component))
    conn.close()
    return df

def plot_pie_chart(df, component):
    if df.empty or df.isnull().values.any():
        return None
    sizes = [df['working_hours'][0], df['maintenance_hours'][0], df['breakdown_hours'][0]]
    labels = ['Working Hours', 'Maintenance Hours', 'Breakdown Hours']
    colors = ['#28a745', '#ffc107', '#dc3545']
    plt.figure(figsize=(5, 5))
    wedges, _, _ = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.3))
    plt.gca().add_artist(plt.Circle((0, 0), 0.60, fc='white'))
    plt.axis('equal')
    plt.title(f'{component} Hour Distribution')
    pie_path = os.path.join(STATIC_DIR, 'hour_distribution_pie.png')
    plt.tight_layout()
    plt.savefig(pie_path)
    plt.close()
    return pie_path

def plot_failure_frequency_bar(machine):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT component, COUNT(*) AS count
        FROM hmr_data WHERE machine = ?
        GROUP BY component
        ORDER BY count DESC
    """, conn, params=(machine,))
    conn.close()
    if df.empty:
        return None
    plt.figure(figsize=(8, 5))
    bars = plt.barh(df['component'], df['count'], color='grey')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, str(int(width)), va='center')
    plt.xlabel("Number of Breakdowns")
    plt.title("Failure Frequency by Component")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    bar_path = os.path.join(STATIC_DIR, 'failure_frequency_bar.png')
    plt.savefig(bar_path)
    plt.close()
    return bar_path

def plot_failure_probability(ttf, component, weibull_cdf, beta, eta):
    sorted_ttf = np.sort(ttf)
    empirical_fp = np.arange(1, len(sorted_ttf)+1) / len(sorted_ttf)
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_ttf, empirical_fp, 'ko', label='Empirical')
    x_vals = np.linspace(0, sorted_ttf.max() * 1.2, 200)
    weibull_curve = weibull_cdf(x_vals, beta, eta)
    plt.plot(x_vals, weibull_curve, 'r-', label='Weibull Fit')
    plt.title(f'Failure Probability: {component}')
    plt.xlabel('Time to Failure (h)')
    plt.ylabel('Failure Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(STATIC_DIR, 'failure_probability_plot.png')
    plt.savefig(path)
    plt.close()
    return path

def plot_gauge(availability, component):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=availability,
        title={'text': f"{component} Availability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccc"},
                {'range': [50, 80], 'color': "#ffe066"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ]
        }
    ))
    path = os.path.join(STATIC_DIR, 'availability_gauge.html')
    fig.write_html(path, include_plotlyjs='cdn')
    return path

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    machine = request.args.get('machine') or request.form.get('machine')
    component = request.args.get('component') or request.form.get('component')
    if not machine:
        return redirect(url_for('welcome'))

    if request.method == 'POST':
        try:
            hmr = float(request.form['hmr_value'])
            desc = request.form['failure_description']
            working = float(request.form['working_hours'])
            maintenance = float(request.form['maintenance_hours'])
            breakdown = float(request.form['breakdown_hours'])
            insert_hmr_value(machine, component, hmr, desc, working, maintenance, breakdown)
            return redirect(url_for('dashboard', machine=machine, component=component))
        except:
            pass

    components = get_components(machine)
    ttf = get_ttf_values(machine, component)

    if len(ttf) < 2:
        return render_template('index.html', selected_machine=machine,
                               selected_component=component, components=components,
                               error="Not enough TTF data available.")

    def weibull_cdf(t, beta, eta):
        return 1 - np.exp(-(t / eta) ** beta)

    sorted_ttf = np.sort(ttf)
    empirical_cdf = np.arange(1, len(sorted_ttf)+1) / len(sorted_ttf)
    popt, _ = curve_fit(weibull_cdf, sorted_ttf, empirical_cdf, p0=[1.5, 100])
    beta, eta = popt
    mttf = eta * gamma(1 + 1 / beta)
    availability = round((mttf / (mttf + 45)) * 100, 2)
    threshold = MTTF_THRESHOLDS.get(component, 9999)
    show_warning = mttf > threshold

    df = fetch_latest_hmr(machine, component)
    plot_pie_chart(df, component)
    plot_failure_probability(ttf, component, weibull_cdf, beta, eta)
    plot_failure_frequency_bar(machine)
    plot_gauge(availability, component)

    return render_template("index.html",
                           selected_machine=machine,
                           selected_component=component,
                           components=components,
                           shape=round(beta, 3),
                           scale=round(eta, 3),
                           mttf=round(mttf, 2),
                           availability=availability,
                           failure_plot=url_for('static', filename='failure_probability_plot.png'),
                           gauge_html=url_for('static', filename='availability_gauge.html'),
                           pie_chart=url_for('static', filename='hour_distribution_pie.png'),
                           bar_chart=url_for('static', filename='failure_frequency_bar.png'),
                           show_warning=show_warning,
                           threshold=threshold)

@app.route('/download_csv')
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM hmr_data", conn)
    conn.close()
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
                     as_attachment=True, download_name='log_data.csv')

if __name__ == '__main__':
    app.run(debug=True)
