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
        return "Dosya seçilmedi"
    
    file = request.files['patient_file']
    
    if file.filename == '':
        return "Dosya seçilmedi"
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Dosyayı oku ve işle
        patient_data = pd.read_csv(filepath, low_memory=False)
        
        # Büyük veri seti için aynı ön işleme adımlarını uygula
        clinic_data = patient_data.loc[:, patient_data.columns[:31]]
        genomic_data = patient_data.loc[:, patient_data.columns[31:]]

        for column in clinic_data.columns:
            if clinic_data[column].isnull().any():
                clinic_data[column] = clinic_data[column].fillna(clinic_data[column].mode()[0])
        
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
        
        for col in genomic_data.columns:
            genomic_data[col] = np.where(genomic_data[col] != '0', 1, 0)
        genomic_data = genomic_data.astype(int)

        patient_processed_data = pd.concat([clinic_data.drop(columns=ordinal_cols + nominal_cols), ordinal_df, nominal_df, genomic_data], axis=1)
        patient_processed_data[['age_at_diagnosis', 'nottingham_prognostic_index']] = patient_processed_data[['age_at_diagnosis', 'nottingham_prognostic_index']].apply(np.ceil)
        patient_processed_data.drop(['patient_id', 'cancer_type'], axis=1, inplace=True)
        
        # Veriyi normalle
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        x_patient = scaler.transform(patient_processed_data.select_dtypes(include=np.number))
        
        # PCA uygulaması
        pca = pickle.load(open('models/pca.pkl', 'rb'))
        principal_components_patient = pca.transform(x_patient)
        principal_df_patient = pd.DataFrame(data=principal_components_patient)
        
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
            death_probabilities = model.predict_proba(principal_df_patient)[:, 1]  # Ölüm olasılıklarını alır
            
            # Sonuçları hazırlama
            result_df = patient_data[['patient_id', 'age_at_diagnosis', 'death_from_cancer']].copy()
            result_df['Death Probability'] = death_probabilities
            result_html = result_df.to_html(index=False)

            # Doğruluk oranlarını yükle
            accuracies = pickle.load(open('models/accuracies.pkl', 'rb'))
            model_name = model_filename.split('.')[0]
            accuracy = accuracies.get(model_name, "Bilinmiyor")
            
            # Doğruluk oranını yüzde formatında göster
            accuracy_percent = f"{accuracy:.2f}%"
            
            return render_template('results.html', tables=result_html, model_choice=model_choice, accuracy=accuracy_percent)
    
    return "Dosya işleme başarısız"

if __name__ == '__main__':
    app.run(debug=True)
