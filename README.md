# Plant Disease Detection using CNN and Transfer Learning

Upload a leaf photo, and the model will predict the disease.
## 🌐 Live Demo
https://plant-disease-detection-ogprsguqebssdnrmd9fma6.streamlit.app

## Project overview
- Model: MobileNetV2 (ImageNet pretrained), transfer learning
- Dataset: PlantVillage (15 classes, ~20,600 images)
- Dataset Link : https://www.kaggle.com/datasets/emmarex/plantdisease
- Framework: TensorFlow/Keras, Streamlit (app)

## Results
- Test accuracy: **88.18%**
- Weighted F1: **0.88**

## Limitations

- PlantVillage images are captured in a lab setting with plain backgrounds, so accuracy may be lower on real field photos.
- Tomato early blight, bacterial spot, and septoria are often confused with each other because their lesions look visually similar.

## Files
- `app.py`: Streamlit app
- `plant_disease_model.keras`: trained model
- `class_names.json`: class labels
- `Plant_Disease_Detection.ipynb`: training notebook

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
