import pymysql
import numpy as np
import json
import os
from flask import Flask, render_template, request, flash, redirect, url_for,send_from_directory,jsonify
from werkzeug.utils import secure_filename
from attribute import extract_features
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Kiểm tra file hợp lệ
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
DATASET_DIR = "Dataset\\"
@app.route("/dataset/<path:filename>")
def serve_audio(filename):
    try:
        return send_from_directory(DATASET_DIR, filename)
    except FileNotFoundError:
        flash(f"File âm thanh {filename} không tìm thấy")
        return redirect(url_for("index"))

# Trang chủ
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Kiểm tra file được tải lên
        if "file" not in request.files:
            flash("Không có file được chọn")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("Chưa chọn file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Trích xuất đặc trưng từ file đầu vào
            try:
                results = extract_features(filepath)
                show_waveform(filepath, save_path="static/waveform.png")
                # Xóa file tạm
                os.remove(filepath)

                return render_template("index.html", results=results)
            except Exception as e:
                flash(f"Lỗi xử lý file: {e}")
                return redirect(request.url)
        else:
            flash("File không hợp lệ. Vui lòng chọn file .wav")
            return redirect(request.url)
    return render_template("index.html", results=None)
@app.route("/compare", methods=["POST"])
def compare():
    if "file" not in request.files:
        return jsonify({"error": "Không có file được chọn"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            results = extract_features(filepath)

            os.remove(filepath)

            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": f"Lỗi xử lý file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File không hợp lệ. Vui lòng chọn file .wav"}), 400
if __name__ == "__main__":
    app.run(debug=True)