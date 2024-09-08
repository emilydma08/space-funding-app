from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/donate')
def donate():
    return render_template('donate.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/categories')
def categories():
    return render_template('categories.html')

if __name__ == '__main__':
    app.run(debug=True)