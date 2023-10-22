from flask import Flask
from flask import request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import commons
import lenet5commons
import efficientnet5commons
import resnet50commons

datasets = {
    'homemade': commons.homemade_class_names,
    'kaggle': commons.kaggle_class_names
}

models = {
    'efficientnet5': {
        'homemade': efficientnet5commons.model_homemade,
        'kaggle': efficientnet5commons.model_kaggle,
        'preprocess': efficientnet5commons.preprocess_image
    },
    'lenet5': {
        'homemade': lenet5commons.model_homemade,
        'kaggle': lenet5commons.model_kaggle,
        'preprocess': lenet5commons.preprocess_image
    },
    'resnet50': {
        'homemade': resnet50commons.model_homemade,
        'kaggle': resnet50commons.model_kaggle,
        'preprocess': resnet50commons.preprocess_image
    }
}

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if (request.args.get('bytesFormat') == 'true'):
        image_stream = BytesIO(request.data)
    elif 'image' in request.files.keys():
        image_stream = request.files['image']
    else:
        return {'status':400,'desc':'Missing input image'}, 400

    try:
        img = Image.open(image_stream)
    except:
        return {'status':400,'desc':'Invalid input image'}, 400

    # The most appropriate model for ingame usage
    dataset = 'homemade'
    model = 'resnet50'
    number_of_predictions = 5

    if ('dataset' in request.args):
        if (request.args.get('dataset') in datasets):
            dataset = request.args.get('dataset')
        else:
            return {'status':400,'desc':'Invalid param dataset'}

    if ('model' in request.args):
        if (request.args.get('model') in models):
            model = request.args.get('model')
        else:
            return {'status':400,'desc':'Invalid param model'}
    
    if ('numberOfPredictions' in request.args):
        if (request.args.get('numberOfPredictions').isdigit() and int(request.args.get('numberOfPredictions')) >= 1 and int(request.args.get('numberOfPredictions')) <= 10):
            number_of_predictions = int(request.args.get('numberOfPredictions'))
        else:
            return {'status':400,'desc':'Invalid param numberOfPredictions'}

    img = img.convert('RGB')

    return commons.predict_biomes(models[model][dataset], datasets[dataset], models[model]['preprocess'](img), number_of_predictions)
