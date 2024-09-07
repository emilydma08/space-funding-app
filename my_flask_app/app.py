from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('PLACEHOLDER.html')

@app.route('/stories')
def stories():
    return render_template('PLACEHOLDER.html')

@app.route('/lessons')
def lessons():
    return render_template('PLACEHOLDER.html')

@app.route('/donate')
def pricing():
    return render_template('PLACEHOLDER.html')

if __name__ == '__main__':
    app.run(debug=True)