# BiomeANN - API

This repository contains the code and multiple models used by the API to classify Minecraft biomes with an image.

## Run

By default the server will run on the port 5000.

```
FLASK_APP=app flask run
```

## Usage

A single route is available `/predict`.

`curl -X POST -F 'image=@samples/dark_forest.png' http://localhost:5000/predict`

Mutliple query arguments can be given to configure the API output.

 - model: 'efficientnet5', 'lenet5', 'resnet50' - Specify the model you wish to use.
 - dataset: 'homemade', 'kaggle' - Specify the dataset.
 - numberOfPredictions: Int between 1 and 10 - The maximum number of predictions you want.
 - bytesFormat: 'true' - Tell the server to expect an image in binary format directly in the body.

Example :

`curl -X POST -F 'image=@samples/desert.png' 'http://localhost:5000/predict?model=lenet5&dataset=kaggle&numberOfPredictions=3'`

```json
[
  {
    "biome": "desert",
    "proba": "0.8017"
  },
  {
    "biome": "desert_hills",
    "proba": "0.1542"
  },
  {
    "biome": "desert_lakes",
    "proba": "0.0245"
  }
]
```

