from keras.models import load_model

from flask import Flask, request, render_template

app = Flask(__name__)

# load model
model = load_model('models/model.h5')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=300)

def pred_sent(text):
    tokenizer.fit_on_texts([text])
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    if prediction == 0:
        return "Postive"
    else:
        return "Negative"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    text = request.form['Tweet']
    text = str(text)
    output = pred_sent(text)
    return render_template('index.html', prediction_text='Sentiment is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=False)