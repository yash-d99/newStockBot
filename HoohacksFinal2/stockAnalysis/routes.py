from flask import Flask, render_template, request, url_for
from stockAnalysis import app
from stockAnalysis.models import stockInput


@app.route('/')
@app.route('/home')
def home():
  return render_template("default.html")

@app.route('/results', methods=['POST'])
def getStuffFromWebsite():
  if request.method == "POST":
    nasdaqsymbol = request.form['symbol']
    predictions = stockInput(nasdaqsymbol)
    return render_template("results.html", predictions=predictions)
  return render_template("default.html")