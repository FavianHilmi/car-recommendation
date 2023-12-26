from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, static_url_path='/static')

# Read and preprocess the data
data = pd.read_csv('CARS_1.csv')

data = data.dropna(subset=['engine_displacement', 'seating_capacity', 'ending_price'])

# Preprocessing: Scaling the numerical columns
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaled_data = scaler.fit_transform(data[['engine_displacement', 'seating_capacity', 'ending_price']])
data[['engine_displacement', 'seating_capacity', 'ending_price']] = scaled_data

# Calculate similarity_matrix using cosine similarity
feature_matrix = data[['engine_displacement', 'seating_capacity', 'ending_price']]
similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)

@app.route('/')
def index():
    return render_template('index.html')  # Tampilan awal dengan form input

def hasil_prediksi(row):
    engine_displacement = row['engine_displacement']
    seating_capacity = row['seating_capacity']
    ending_price = row['ending_price']

    user_input = scaler.transform([[engine_displacement, seating_capacity, ending_price]])

    similarities = cosine_similarity(user_input, feature_matrix)
    # Misalnya, Anda ingin mengembalikan nilai similarity
    return similarities[0][0]

# Buat kolom baru dengan nilai similarity/rekomendasi untuk setiap baris pada dataset
data['modified'] = data.apply(hasil_prediksi, axis=1)

@app.route('/hasil_prediksi', methods=['POST'])
def hasil_prediksi():
    ending_price = float(request.form['ending_price'])
    engine_displacement = float(request.form['engine_displacement'])
    seating_capacity = float(request.form['seating_capacity'])
    transmission_type_input = request.form['transmission_type']

    # Normalizing user input
    user_input = scaler.transform([[engine_displacement, seating_capacity, ending_price]])

    # Calculate similarity of user input with existing data
    similarities = cosine_similarity(user_input, feature_matrix)
    similar_indices = similarities.argsort()[0][-6:-1][::-1]
    
    # Retrieve the transmission types for the similar cars
    similar_transmission_types = data.iloc[similar_indices]['transmission_type'].values

    # # Adjust similarity scores based on transmission priority
    # for i, similar_transmission in enumerate(similar_transmission_types):
    #     if similar_transmission == transmission_type_input:
    #         similarities[0][similar_indices[i]] *= 1.2  # Increase the similarity score
    #     else:
    #         similarities[0][similar_indices[i]] *= 0.8  # Decrease the similarity score
    
    similar_cars_data = data.iloc[similar_indices][['car_name', 'engine_displacement', 'seating_capacity', 'ending_price', 'transmission_type']]
    similar_cars_data[['engine_displacement', 'seating_capacity', 'ending_price']] = scaler.inverse_transform(similar_cars_data[['engine_displacement', 'seating_capacity', 'ending_price']])
    # similar_cars_data['Similarity Score'] = similarities[0][similar_indices]
    similar_cars_data['transmission_type'] = data.iloc[similar_indices]['transmission_type'].values
    
    return render_template('result.html', similar_cars_data=similar_cars_data.values)

if __name__ == '__main__':
    app.run(debug=True)
