# Plant Disease Detection using CNN and Transfer Learning

Upload a leaf photo, and the model predicts the plant disease, shows treatment suggestions, and highlights the image regions the model focused on (Grad-CAM).

🌐 **Live demo:** https://plant-disease-detection-ogprsguqebssdnrmd9fma6.streamlit.app

## Features
- Disease prediction for 15 classes (Potato, Tomato, Pepper bell)
- Top 3 predictions with confidence scores
- Confidence threshold (70%): low-confidence results show a warning instead of a guess
- Simple leaf check: images without enough leaf-like green colour are rejected
- Treatment suggestions for each disease (general guidance)
- Grad-CAM heatmap showing where the model looked

## Project overview
- **Model:** MobileNetV2 (ImageNet pretrained), transfer learning, custom classification head
- **Dataset:** PlantVillage (15 classes, ~20,600 images)
-  **Dataset link**:https://www.kaggle.com/datasets/emmarex/plantdisease?select=PlantVillage
- **Framework:** TensorFlow/Keras (training), Streamlit (web app)
- **Training environment:** Google Colab (T4 GPU)

## Training details
- Data split: 80% train, 10% validation, 10% test
- Augmentation: random flip, rotation, zoom
- Phase 1: base model frozen, only the classifier head trained (Adam, lr 1e-3)
- Early stopping and best-model checkpoint used
- Trained for 10 epochs

## Results
- **Test accuracy:** 88.18%
- **Weighted F1-score:** 0.88
- Best classes: Potato early blight, Tomato Yellow Leaf Curl Virus, Pepper bell healthy (F1 above 0.95)
- Weakest class: Tomato early blight (recall 0.59)

## Limitations
- PlantVillage images are captured in a lab setting with plain backgrounds, so accuracy may be lower on real field photos.
- Tomato early blight, bacterial spot, and septoria are often confused with each other because their lesions look visually similar.
- The model only knows the 15 trained classes. Objects outside these classes (for example, furniture, posters, or people) can still get a prediction.
- The green-colour leaf check is a simple rule. It can accept non-leaf images with a lot of green (such as posters), and can reject real leaves on very dark or colourful backgrounds.
- Stock or internet photos with white backgrounds can get low confidence or wrong predictions.
- Treatment suggestions are general guidance and are not based on the dataset. Always consult a local agriculture expert before using any chemical.

## Future work
- Add a "not a leaf" class to train a proper leaf detector
- Train on real field photos (for example, PlantDoc dataset)
- Compare MobileNetV2 with EfficientNetB0
- Replace the colour rule with a stronger rejection step for unknown inputs

## Files
- `app.py`: Streamlit app (prediction, threshold, treatment, Grad-CAM)
- `plant_disease_model.keras`: trained model
- `class_names.json`: class labels
- `requirements.txt`: Python dependencies
- `Plant_Disease_Detection.ipynb`: training notebook

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `plant_disease_model.keras` and `class_names.json` are in the same folder as `app.py`.

## Author
Maaz Safi (BS Data Science, 5th semester)
