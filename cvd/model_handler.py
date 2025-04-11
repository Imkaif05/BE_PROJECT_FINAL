# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np

# # Load the pre-trained model
# MODEL_PATH = 'C:\Users\kaifm\OneDrive\Desktop\Project BE-2025\Heart-Care\cvd\model.h5'  # Update with the correct model path
# model = tf.keras.models.load_model(MODEL_PATH)
# IMG_SIZE = 256  # Adjust based on training settings

# # Class labels
# CLASS_LABELS = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# # Heart disease probability mapping
# HEART_DISEASE_PROBABILITIES = {
#     'cataract': (0, 2), 
#     'diabetic_retinopathy': (30, 40), 
#     'glaucoma': (15, 25), 
#     'normal': (0, 2)
# }

# # Function to preprocess image
# def prepare_image(img_path):
#     img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize
#     return img_array

# # Function to predict image class
# def predict_image(img_path):
#     img_array = prepare_image(img_path)
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     predicted_label = CLASS_LABELS[predicted_class]
#     predicted_probability = predictions[0][predicted_class] * 100

#     # Get heart disease probability
#     prob_range = HEART_DISEASE_PROBABILITIES[predicted_label]
#     heart_disease_probability = np.random.uniform(prob_range[0], prob_range[1])

#     return {
#         'predicted_class': predicted_label,
#         'prediction_probability': f'{predicted_probability:.2f}%',
#         'heart_disease_probability': f'{heart_disease_probability:.2f}%'
#     }
