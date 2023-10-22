import torch

homemade_class_names = {
    0: 'badlands',
    1: 'bamboo_jungle',
    2: 'beach',
    3: 'birch_forest',
    4: 'cherry_grove',
    5: 'dark_forest',
    6: 'desert',
    7: 'eroded_badlands',
    8: 'flower_forest',
    9: 'forest',
    10: 'frozen_peaks',
    11: 'giant_spruce_taiga',
    12: 'ice_spikes',
    13: 'jungle',
    14: 'mountains',
    15: 'mushroom_fields',
    16: 'plains',
    17: 'savanna',
    18: 'snowy_beach',
    19: 'snowy_mountains',
    20: 'snowy_plains',
    21: 'snowy_taiga',
    22: 'stone_shore',
    23: 'sunflower_plain',
    24: 'swamp',
    25: 'taiga',
    26: 'taiga_hills',
    27: 'tall_birch_forest',
    28: 'windstep_forest',
    29: 'windstep_savanna',
    30: 'wooded_badland_plateau',
}

kaggle_class_names = {
    0: 'plains',
    1: 'frozen_ocean',
    2: 'frozen_river',
    3: 'snowy_tundra',
    4: 'sunflower_plains',
    5: 'snowy_mountains',
    6: 'desert_lakes',
    7: 'gravelly_mountains',
    8: 'flower_forest',
    9: 'taiga_mountains',
    10: 'tall_birch_hills',
    11: 'dark_forest_hills',
    12: 'snowy_taiga_mountains',
    13: 'beach',
    14: 'modified_gravelly_mountains',
    15: 'desert_hills',
    16: 'wooded_hills',
    17: 'taiga_hills',
    18: 'desert',
    19: 'jungle',
    20: 'jungle_hills',
    21: 'snowy_beach',
    22: 'birch_forest',
    23: 'birch_forest_hills',
    24: 'dark_forest',
    25: 'mountains',
    26: 'snowy_taiga',
    27: 'snowy_taiga_hills',
    28: 'giant_tree_taiga',
    29: 'giant_tree_taiga_hills',
    30: 'wooded_mountains',
    31: 'savanna',
    32: 'savanna_plateau',
    33: 'badlands',
    34: 'wooded_badlands_plateau',
    35: 'badlands_plateau',
    36: 'forest',
    37: 'lukewarm_ocean',
    38: 'taiga',
    39: 'swamp',
    40: 'river'
}

def predict_biomes(model, class_names, image, topk=10):
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        topk_probs, topk_labels = torch.topk(probabilities, topk, dim=1)
    
    predicted_labels = topk_labels[0].cpu().numpy()
    predicted_biomes = [class_names.get(label, "unknown") for label in predicted_labels]
    
    topk_probs = topk_probs[0].cpu().numpy()

    prediction = []
    for i in range(len(predicted_biomes)):
        prediction.append({ "biome": predicted_biomes[i], "proba": str(round(topk_probs[i], 4)) })
    
    return prediction
