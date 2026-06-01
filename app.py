from flask import Flask, render_template, send_from_directory
import sqlite3

app = Flask(__name__)

@app.route('/detected_vehicles/<path:filename>')
def vehicle_image(filename):
    return send_from_directory(
        'detected_vehicles',
        filename
    )
DB_PATH = "vehicles.db"


def get_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT plate, timestamp, source, image_path
        FROM detections
        ORDER BY id DESC
    """)

    data = cursor.fetchall()
    conn.close()

    return data


@app.route("/")
def home():
    data = get_data()
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)