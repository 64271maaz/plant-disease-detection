# Plant Disease Detection using CNN and Transfer Learning

Upload a leaf photo, and the model will predict the disease.
## 🌐 Live Demo
https://plant-disease-detection-ogprsguqebssdnrmd9fma6.streamlit.app

## Project overview
- Model: MobileNetV2 (ImageNet pretrained), transfer learning
- Dataset: PlantVillage (15 classes, ~20,600 images)
- Framework: TensorFlow/Keras, Streamlit (app)

## Results
- Test accuracy: **88.18%**
- Weighted F1: **0.88**

## Limitations
- PlantVillage images lab-controlled hain (plain background). Real field photos par accuracy kam ho sakti hai.
- Tomato early blight, bacterial spot aur septoria aapas mein confuse hote hain.

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
