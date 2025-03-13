import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Veriyi ön işleme fonksiyonu
def preprocess_data(df):
    # Klinik verileri ayır
    clinic_data = df.loc[:, df.columns[:31]]
    genomic_data = df.loc[:, df.columns[31:]]
    
    # Klinik verilerde boş değerleri mod ile doldur
    for column in clinic_data.columns:
        clinic_data[column] = clinic_data[column].fillna(clinic_data[column].mode()[0])
    
    # Ordinal ve nominal verileri kodla
    ordinal_cols = [
        'cellularity', 'cancer_type_detailed', 'type_of_breast_surgery',
        'her2_status_measured_by_snp6', 'pam50_+_claudin-low_subtype',
        'tumor_other_histologic_subtype', 'integrative_cluster',
        '3-gene_classifier_subtype', 'death_from_cancer'
    ]
    ordinal_df = clinic_data[ordinal_cols].apply(LabelEncoder().fit_transform)
    
    nominal_cols = [
        'er_status_measured_by_ihc', 'er_status', 'her2_status', 
        'inferred_menopausal_state', 'primary_tumor_laterality', 
        'pr_status', 'oncotree_code'
    ]
    nominal_df = pd.get_dummies(clinic_data[nominal_cols], drop_first=True)
    
    # Genomik veriler
    for col in genomic_data.columns:
        genomic_data[col] = np.where(genomic_data[col] != '0', 1, 0)
    genomic_data = genomic_data.astype(int)
    
    # Verileri birleştir
    df = pd.concat([clinic_data.drop(columns=ordinal_cols + nominal_cols), ordinal_df, nominal_df, genomic_data], axis=1)
    
    # Yaşı ve prognostik indeksi yuvarla
    df[['age_at_diagnosis', 'nottingham_prognostic_index']] = df[['age_at_diagnosis', 'nottingham_prognostic_index']].apply(np.ceil)
    
    # İlgisiz sütunları düş
    df.drop(['patient_id', 'cancer_type'], axis=1, inplace=True)
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'patient_file' not in request.files:
        return "No file part"
    
    file = request.files['patient_file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Dosyayı oku ve işle
        patient_data = pd.read_csv(filepath, low_memory=False)
        processed_data = preprocess_data(patient_data)
        
        # Veriyi normalle
        numeric_df = processed_data.select_dtypes(include=np.number)
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        x = scaler.transform(numeric_df)
        
        # PCA uygulaması
        pca = pickle.load(open('models/pca.pkl', 'rb'))
        principal_components = pca.transform(x)
        principal_df = pd.DataFrame(data=principal_components)
        
        # Modeli yükle ve tahmin yap
        model_choice = request.form['model_choice']
        model_mapping = {
            'Logistic Regression': 'logistic_regression.pkl',
            'K-Nearest Neighbors': 'knn_model.pkl',
            'Random Forest': 'random_forest.pkl'
        }
        
        model_filename = model_mapping.get(model_choice)
        if model_filename:
            model = pickle.load(open(f'models/{model_filename}', 'rb'))
            predictions = model.predict(principal_df)
            
            # Doğruluk oranını yükle
            accuracy = pickle.load(open(f'models/{model_filename.split(".")[0]}_accuracy.pkl', 'rb'))
            
            # Sonuçları hazırlama
            result_df = patient_data.iloc[:, :4].copy()  # İlk 4 sütunu al
            result_df['Prediction'] = predictions
            result_html = result_df.to_html(classes='dataframe')
            
            return render_template('results.html', tables=result_html, model_choice=model_choice, accuracy=accuracy)
    
    return "File processing failed"

if __name__ == '__main__':
    app.run(debug=True)
