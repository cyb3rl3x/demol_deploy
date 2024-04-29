# app.py
import streamlit as st
import pickle
import numpy as np

# Carregar o modelo treinado
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Título do aplicativo
st.title('Iris Species Prediction App')

# Receber inputs do usuário
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Botão para fazer a previsão
if st.button('Predict'):
    # Criar array com inputs
    test_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # Fazer a previsão
    prediction = model.predict(test_array)
    species = data['target_names'][prediction][0]
    # Mostrar a previsão
    st.write(f'The predicted species is: {species}')


