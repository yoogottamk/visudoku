from flask import Flask, request, render_template
import base64 as b64
import ast
import os

import visudoku

CACHE = []

def to_b64(all_stages):
    """
    Converts list of images (np arrays) to base64 encoded images
    """

    ret = []

    for stage in all_stages:
        img_b64 = b64.b64encode(visudoku.cv.imencode(".jpg", stage["img"])[1]).decode("utf-8")
        ret.append((stage["label"], img_b64))

    return ret

def clean_sudoku_sol(sudoku_sol):
    for ch in [" ", "+", "-", "|", "\n"]:
        sudoku_sol = sudoku_sol.replace(ch, "")

    return sudoku_sol

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = './uploads'

@app.route("/", methods=["GET", "POST"])
def homepage():
    return render_template("homepage.html")

@app.route("/solve", methods=["GET", "POST"])
def solve():
    img = request.files['image']
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], img.filename)

    img.save(img_path)

    cache_found = False

    for cached in CACHE:
        if img_path == cached["path"]:
            solution, success, img_stages, nums = cached["result"]
            cache_found = True
            break

    if not cache_found:
        solution, success, img_stages, nums = visudoku.solve_visudoku(img_path)
        CACHE.append({ "path": img_path, "result": [solution, success, img_stages, nums]})

    return render_template("solve.html", solution=clean_sudoku_sol(solution), success=success, all_stages=to_b64(img_stages), nums=" ".join(nums))

@app.route("/fix", methods=["GET", "POST"])
def fix():
    inp = request.form['inp']
    all_stages = ast.literal_eval(request.form['all_stages'])

    solution, success = visudoku.solve(inp)

    return render_template("solve.html", solution=clean_sudoku_sol(solution), success=success, all_stages=all_stages, nums=inp)

app.run(host="127.0.0.1", port=8080, debug=True)
