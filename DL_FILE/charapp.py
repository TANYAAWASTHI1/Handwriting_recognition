# import streamlit as st
# import numpy as np
# from PIL import Image, ImageOps
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import joblib
#
# # Set the Page Title
# st.set_page_config(
#     page_title="Hindi Character Recognition",
#     page_icon="🔎"
# )
#
# # Hindi character list
# hindi_character = 'क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह ॠ त्र ज्ञ ० १ २ ३ ४ ५ ६ ७ ८ ९'.split()
#
# # Load the trained model
# @st.cache(allow_output_mutation=True)
# def load_model():
#     return joblib.load('best_svc_classifier_model.joblib')
#
# model = load_model()
#
# # Load and preprocess the image
# def load_and_prep(file):
#     img = Image.open(file).convert('L')  # Convert image to grayscale
#     img = ImageOps.invert(img)  # Invert image colors
#     img = img.resize((32, 32))  # Resize image
#     img = np.array(img).flatten()  # Flatten image
#     return img
#
# # Get top n predictions
# def get_n_predictions(pred_prob, n):
#     top_n_max_idx = np.argsort(pred_prob)[::-1][:n]  # Get index of top n predictions
#     top_n_max_val = list(pred_prob[top_n_max_idx])  # Get actual top n predictions
#     top_n_class_name = [hindi_character[i] for i in top_n_max_idx]  # Get corresponding Hindi characters
#     return top_n_class_name, top_n_max_val
#
# # Streamlit app
# st.title("Hindi Character Recognition")
#
# # Upload image
# file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
#
# if file is not None:
#     img = load_and_prep(file)
#
#     # Display uploaded image
#     st.image(file, caption='Uploaded Image', use_column_width=True)
#
#     # Make prediction
#     if st.button('Predict'):
#         pred_prob = model.predict_proba([img])
#         n = st.slider('Select Top N Predictions', min_value=1, max_value=len(hindi_character), value=3, step=1)
#
#         class_name, confidence = get_n_predictions(pred_prob[0], n)
#
#         # Display top N predictions
#         st.write("Top Predictions:")
#         for i in range(n):
#             st.write(f"{class_name[i]}: {confidence[i]*100:.2f}%")
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Set the Page Title
st.set_page_config(
    page_title="Hindi Character Recognition",
    page_icon="🔎"
)

# Hindi character list
hindi_character = 'क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह ॠ त्र ज्ञ ० १ २ ३ ४ ५ ६ ७ ८ ९'.split()

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('best_svc_classifier_model.joblib')

model = load_model()

# Load and preprocess the image
def load_and_prep(file):
    # Load and preprocess the image
        if file is not None:
            img = Image.open(file).convert('L')  # Convert image to grayscale
            img = ImageOps.invert(img)  # Invert image colors
            img = img.resize((17, 17))  # Resize image to match the expected input size (17x17)
            img = np.array(img).flatten()  # Flatten image
            # Add padding to the flattened image to make it 293 features
            img = np.pad(img, (0, 293 - len(img)), mode='constant')
            return img
        else:
            st.error('No file uploaded. Please upload an image.')
            return None


# img = Image.open(file).convert('L')  # Convert image to grayscale
    # img = ImageOps.invert(img)  # Invert image colors
    # img = img.resize((32, 32))  # Resize image
    # img = np.array(img).flatten()  # Flatten image
    # return img

# Get top n predictions
def get_n_predictions(decision, n):
    top_n_max_idx = np.argsort(decision)[::-1][:n]  # Get index of top n predictions
    top_n_max_val = list(decision[top_n_max_idx])  # Get actual top n predictions
    top_n_class_name = [hindi_character[i] for i in top_n_max_idx]  # Get corresponding Hindi characters
    return top_n_class_name, top_n_max_val

# Streamlit app
st.title("Hindi Character Recognition")

# Upload image
file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if file is not None:
    img = load_and_prep(file)

    # Display uploaded image
    st.image(file, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    if st.button('Predict'):

        img = img.reshape(1, -1)  # Reshape img to be a 2D array with a single row

        decision = model.decision_function(img)
        n = st.slider('Select Top N Predictions', min_value=1, max_value=5, value=3, step=1)

        class_name, confidence = get_n_predictions(decision[0], n)

        # Display top N predictions
        st.write("Top Predictions:")
        for i in range(n):
            st.write(f"{class_name[i]}: {confidence[i]:.2f}")







