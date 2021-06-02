from flask import Flask, current_app, send_file, jsonify, request
import os

from src.test import main as get_anime

app = Flask(__name__)
app._static_folder = os.path.abspath("static/")


@app.route('/')
def main():
    return current_app.send_static_file('index.html')


@app.route('/anime')
def gen_anime():
    eyes = request.args.get('eyes')
    hair = request.args.get('hair')
    anime = get_anime(eyes=eyes, hair=hair)
    return send_file(anime, mimetype='image/gif')


app.run()
