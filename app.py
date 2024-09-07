from flask import Flask, render_template

app = Flask(__name__)

@app.route('/templates/index.html')
def home():
    return render_template('index.html')

@app.route('/templates/donate.html')
def about():
    return render_template('donate.html')

@app.route('/templates/stories.html')
def stories():
    return render_template('stories.html')

@app.route('/templates/LessonPage1.html')
def lessons():
    return render_template('LessonPage1.html')

@app.route('/templates/pricing.html')
def pricing():
    return render_template('pricing.html')

if __name__ == '__main__':
    app.run(debug=True)