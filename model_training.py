import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Model dosyalarının kaydedileceği dizini oluştur
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Veriyi yükle
df = pd.read_csv("C:/Users/yekre/OneDrive/Masaüstü/METABRIC_RNA_Mutation.csv", low_memory=False)

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

# Veriyi işle
processed_data = preprocess_data(df)

# Veriyi normalle
numeric_df = processed_data.select_dtypes(include=np.number)
scaler = MinMaxScaler().fit(numeric_df)
x = scaler.transform(numeric_df)
y = processed_data['death_from_cancer']

# PCA uygulaması
pca = PCA(n_components=50)  # n_components değerini düşürdük
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components)

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(principal_df, y, test_size=0.50, random_state=42)  # test_size=0.50 ayarlandı

# StratifiedKFold kullanarak modeli 10 tekrar (k=10) ile eğit
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Modelleri eğit ve kaydet
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, C=0.01, random_state=42),  # C parametresini azaltarak regularization ekledik
    'knn_model': KNeighborsClassifier(n_neighbors=10),  # n_neighbors azalttık
    'random_forest': RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=42)  # n_estimators 1000 yapıldı
}

accuracies = {}

for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf)
    accuracy = np.mean(cv_scores)
    accuracies[model_name] = accuracy * 100  # Yüzdelik olarak kaydet
    model.fit(X_train, y_train)
    with open(f'{model_dir}/{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)

# Scaler ve PCA nesnelerini kaydet
with open(f'{model_dir}/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
with open(f'{model_dir}/pca.pkl', 'wb') as file:
    pickle.dump(pca, file)

# Doğruluk oranlarını kaydet
with open(f'{model_dir}/accuracies.pkl', 'wb') as file:
    pickle.dump(accuracies, file)

print("Modeller, Scaler, PCA nesneleri ve doğruluk oranları başarıyla eğitildi ve kaydedildi.")

