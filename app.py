from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# load json and create model
json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
tokenizer = Tokenizer(num_words=5000)

def pred_sent(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(loaded_model.predict(tw).round().item())
    if prediction == 0:
        return 'Positive'
    return 'Negative'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.values()
    output = pred_sent(text)
    return render_template('index.html', prediction_text='The tweet sentiment is {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)