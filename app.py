import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Aplikasi ---
st.set_page_config(page_title="Aplikasi Klasifikasi Data Pasien", layout="wide")


# Tambahkan Pilihan Tema di Sidebar
theme = st.sidebar.radio("Pilih Tema Tampilan:", ["Light Mode", "Dark Mode"])

# Styling Custom Berdasarkan Tema
if theme == "Light Mode":
    st.markdown(
        """
        <style>
        /* Styling Header */
        .main-header {
            font-size: 45px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Light Mode Sidebar */
        .sidebar .sidebar-content {
            background-color: #ffffff;
            color: black;
        }

        /* Light Mode Button */
        div.stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }

        /* Footer */
        .footer {
            background-color: #ffffff;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
elif theme == "Dark Mode":
    st.markdown(
        """
        <style>
        /* Styling Header */
        .main-header {
            font-size: 45px;
            font-weight: bold;
            color: #FFD700;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        /* Dark Mode Sidebar */
        .sidebar .sidebar-content {
            background-color: #333333;
            color: white;
        }

        /* Dark Mode Button */
        div.stButton > button {
            background-color: #FFD700;
            color: black;
        }
        div.stButton > button:hover {
            background-color: #FFA500;
        }

        /* Footer */
        .footer {
            background-color: #333333;
            color: #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Tambahkan Informasi Pengembang
st.sidebar.markdown("## 👨‍💻 Tentang Pengembang")
st.sidebar.markdown("""
- Nama: Fadhaeil Rizki Mahwendi
- NIM: 211220095
- Email: [Email ke saya](mailto:211220095@unmuhpnk.ac.id)
- Github : [Link Github saya](https://github.com/FadhaeilRizkiMahwendi)
""")



# --- File untuk Riwayat Prediksi ---
log_file = "prediction_log.csv"

# --- Load Dataset ---
dataset_path = "Classification.csv"
model_path = "model.pkl"

if not os.path.exists(dataset_path):
    st.error(f"File {dataset_path} tidak ditemukan. Pastikan file berada di lokasi yang benar.")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

data = load_data(dataset_path)

# --- Preprocessing Data ---
def preprocess_data(data, target_col):
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object' and col != target_col:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    if data[target_col].dtype in ['float64', 'int64'] and len(data[target_col].unique()) > 10:
        st.warning(f"Kolom target '{target_col}' terdeteksi sebagai data kontinu. Akan dikonversi menjadi kategori.")
        data[target_col] = pd.qcut(data[target_col], q=4, labels=['low', 'medium', 'high', 'very high'])

    return data, label_encoders

# --- Fungsi Training Model ---
def train_model(data, target_col):
    data, label_encoders = preprocess_data(data, target_col)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    feature_names = list(X_train.columns)
    return model, accuracy, feature_names, label_encoders, y_test, y_pred

# --- Fungsi Prediksi ---
def predict_new_data(model_path, input_data):
    with open(model_path, "rb") as f:
        saved_data = pickle.load(f)
        model = saved_data["model"]
        feature_names = saved_data["feature_names"]
        label_encoders = saved_data["label_encoders"]
    for col, le in label_encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col])
    input_data = input_data[feature_names]
    return model.predict(input_data)

# --- Logging Prediksi ---
def log_prediction(log_file, input_data, prediction):
    # Tambahkan kolom Prediction ke data input
    input_data["Prediction"] = prediction

    # Buat file log jika belum ada
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(",".join(list(input_data.columns)) + "\n")

    # Simpan data ke file log
    input_data.to_csv(log_file, mode="a", header=False, index=False)


# --- Visualisasi Confusion Matrix ---
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)


# --- Header ---
st.markdown('<div class="main-header">🌟 Aplikasi Klasifikasi Data Pasien</div>', unsafe_allow_html=True)

# --- Navigation Bar ---
menu = st.radio(
    "Navigasi",
    ["Home", "Training Model", "Prediksi Obat", "Riwayat Prediksi", "Optimisasi Model", "Performa Model", "Feature Importance", "Coba Model Lain", "Help"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_menu"
)


# --- Home Page (Visualisasi Dataset) ---
if menu == "Home":
    st.write("## Selamat Datang di Aplikasi Klasifikasi Data Pasien!")
    st.write("""
    Aplikasi ini dirancang untuk membantu Anda melakukan analisis klasifikasi data pasien.
    Anda dapat menggunakan aplikasi ini untuk:
    - Melatih model machine learning menggunakan dataset yang Anda miliki.
    - Melakukan prediksi data pasien baru berdasarkan input tertentu.
    - Melihat riwayat prediksi sebelumnya.
    - Melakukan optimisasi model untuk mendapatkan performa terbaik.
    - Menjelajahi pentingnya fitur-fitur dalam prediksi (Feature Importance).
    - Mencoba algoritma machine learning model lain untuk di coba
    
    Gunakan navigasi di atas untuk mengakses setiap fitur aplikasi. Selamat mencoba! 🎉
    """)

    st.write("### Informasi Dataset yang Digunakan:")

    # Tambahkan dropdown untuk memilih informasi yang ingin dilihat
    options = ["Tampilkan Dataset", "Deskripsi Statistik Dataset"]
    selected_option = st.selectbox("Pilih informasi yang ingin dilihat:", options)

    # Tampilkan dataset atau deskripsi statistik berdasarkan pilihan
    if selected_option == "Tampilkan Dataset":
        st.write("### Dataset:")
        st.dataframe(data)  # Menampilkan seluruh dataset
    elif selected_option == "Deskripsi Statistik Dataset":
        st.write("### Deskripsi Statistik Dataset:")
        st.dataframe(data.describe())  # Menampilkan deskripsi statistik

    # Visualisasi Kolom
    st.write("### Visualisasi Kolom Dataset:")
    selected_column = st.selectbox("Pilih kolom untuk melihat distribusinya:", data.columns)
    st.write(f"#### Histogram untuk Kolom `{selected_column}`:")
    fig, ax = plt.subplots()
    ax.hist(data[selected_column], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f'Distribusi Kolom {selected_column}')
    ax.set_xlabel(selected_column)
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    # Scatterplot
    st.write("### Scatterplot (Hubungan Dua Kolom):")
    col1 = st.selectbox("Pilih kolom X:", data.select_dtypes(include=['float64', 'int64']).columns)
    col2 = st.selectbox("Pilih kolom Y:", data.select_dtypes(include=['float64', 'int64']).columns)
    st.write(f"#### Scatterplot `{col1}` vs `{col2}`:")
    fig, ax = plt.subplots()
    ax.scatter(data[col1], data[col2], alpha=0.7, c='orange', edgecolor='k')
    ax.set_title(f'{col1} vs {col2}')
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    st.pyplot(fig)

    # Korelasi Heatmap
    st.write("### Korelasi Antar Kolom Numerik:")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Heatmap Korelasi')
        st.pyplot(fig)
    else:
        st.write("Data tidak memiliki cukup kolom numerik untuk menghitung korelasi.")

# --- Training Model Page ---
elif menu == "Training Model":
    st.write("## Training Model")
    target_col = st.selectbox("Pilih kolom target (label):", data.columns)

    if st.button("Mulai Training"):
        with st.spinner("Melatih model, mohon tunggu..."):
            # Latih model
            model, accuracy, feature_names, label_encoders, y_test, y_pred = train_model(data, target_col)
            
            # Simpan model yang sudah dilatih
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "feature_names": feature_names, "label_encoders": label_encoders}, f)

            # Tampilkan hasil evaluasi
            st.success(f"Model berhasil dilatih dengan akurasi {accuracy:.2f}.")
            
            # Visualisasi Confusion Matrix
            st.write("### Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            # Visualisasi Feature Importance (jika Random Forest digunakan)
            if isinstance(model, RandomForestClassifier):
                st.write("### Feature Importance:")
                feature_importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": feature_importance
                }).sort_values(by="Importance", ascending=False)

                # Tampilkan tabel
                st.dataframe(importance_df)

                # Tampilkan grafik bar
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)


# --- Prediksi Data Baru Page ---
elif menu == "Prediksi Obat":
    st.write("## 💊Prediksi Obat")

    # Periksa apakah model sudah dilatih
    if not os.path.exists(model_path):
        st.warning("⚠️ Harap lakukan training model terlebih dahulu di halaman **Training Model** sebelum melakukan prediksi.")
    else:
        # Tata Letak dengan Kolom
        col1, col2 = st.columns([1, 2])

        # Input Data di Kolom Kiri
        with col1:
            st.write("### Input Data")

            # Input numerik dengan validasi rentang
            umur = st.slider("Umur", min_value=15, max_value=74, value=36)
            rasio_na_k = st.slider("Rasio Na ke K", min_value=6.27, max_value=38.25, value=17.71)

            # Input kategorikal dengan pilihan valid
            jenis_kelamin = st.selectbox("Jenis Kelamin", options=["F", "M"])
            tekanan_darah = st.selectbox("Tekanan Darah", options=["LOW", "NORMAL", "HIGH"])
            kolesterol = st.selectbox("Kolesterol ", options=["NORMAL", "HIGH"])

            # Tombol Prediksi
            if st.button("Prediksi"):
                # Validasi semua input sebelum melakukan prediksi
                errors = []
                if not (15 <= umur <= 74):
                    errors.append("Umur harus berada dalam rentang 15-74.")
                if not (6.27 <= rasio_na_k <= 38.25):
                    errors.append("Rasio Na ke K harus berada dalam rentang 6.27-38.25.")
                if jenis_kelamin not in ["F", "M"]:
                    errors.append("Jenis Kelamin harus F atau M.")
                if tekanan_darah not in ["LOW", "NORMAL", "HIGH"]:
                    errors.append("Tekanan Darah harus LOW, NORMAL, atau HIGH.")
                if kolesterol not in ["NORMAL", "HIGH"]:
                    errors.append("Kolesterol harus NORMAL atau HIGH.")

                # Tampilkan error jika ada
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    try:
                        # Input valid, lakukan prediksi
                        user_data = pd.DataFrame({
                            "Age": [umur],
                            "Na_to_K": [rasio_na_k],
                            "Cholesterol": [kolesterol],
                            "BP": [tekanan_darah],
                            "Sex": [jenis_kelamin]
                        })

                        # Prediksi
                        prediction = predict_new_data(model_path, user_data)

                        # Logging Prediksi
                        log_prediction(log_file, user_data, prediction)

                        # Simpan hasil prediksi ke session state
                        st.session_state.prediction = prediction[0]
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat prediksi: {e}")

        # Hasil Visualisasi di Kolom Kanan
        with col2:
            st.write("### Hasil Prediksi")
            if "prediction" in st.session_state:
                st.success(f"Hasil Prediksi: {st.session_state.prediction}")
            else:
                st.info("Silakan masukkan data dan klik tombol Prediksi untuk melihat hasil.")

# --- Riwayat Prediksi Page ---
elif menu == "Riwayat Prediksi":
    st.write("## Riwayat Prediksi")
    
    # Periksa apakah file log ada
    if os.path.exists(log_file):
        try:
            # Baca file log
            log_data = pd.read_csv(log_file)

            if log_data.empty:
                st.write("Belum ada riwayat prediksi.")
            else:
                st.dataframe(log_data)
                st.download_button(
                    label="Unduh Riwayat Prediksi",
                    data=log_data.to_csv(index=False),
                    file_name="prediction_log.csv",
                    mime="text/csv"
                )

                # Tombol Hapus Riwayat
                if st.button("Hapus Riwayat Prediksi"):
                    os.remove(log_file)
                    # Inisialisasi file kosong
                    dummy_data = pd.DataFrame(columns=["Age", "Na_to_K", "Cholesterol", "BP", "Sex", "Prediction"])
                    dummy_data.to_csv(log_file, index=False)
                    st.success("Riwayat prediksi berhasil dihapus.")
        except pd.errors.EmptyDataError:
            st.write("Belum ada riwayat prediksi.")
    else:
        st.write("Belum ada riwayat prediksi.")

# --- Optimisasi Model Page ---
elif menu == "Optimisasi Model":
    st.write("## Optimisasi Model")
    st.write("Atur hyperparameter untuk Random Forest dan latih model untuk mendapatkan performa terbaik.")

    # Tambahkan dropdown untuk memilih kolom target
    target_col = st.selectbox("Pilih kolom target (label):", data.columns)

    # Input untuk hyperparameter
    n_estimators = st.slider("Jumlah Pohon (n_estimators):", min_value=10, max_value=200, step=10, value=100)
    max_depth = st.slider("Kedalaman Maksimum (max_depth):", min_value=1, max_value=50, step=1, value=10)

    if st.button("Optimalkan Model"):
        with st.spinner("Melatih model dengan hyperparameter yang dipilih..."):
            data, label_encoders = preprocess_data(data, target_col)
            X = data.drop(columns=[target_col])
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Latih model dengan hyperparameter
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Simpan model terbaik
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "feature_names": list(X.columns), "label_encoders": label_encoders}, f)
            
            st.success(f"Model berhasil dilatih dengan akurasi {accuracy:.2f}. Hyperparameter telah disimpan!")
            st.write(f"### Akurasi Model: {accuracy:.2f}")

            # Visualisasi Confusion Matrix
            st.write("### Confusion Matrix:")
            plot_confusion_matrix(y_test, y_pred)

# --- Performa Model Page ---
elif menu == "Performa Model":
    st.write("## Performa Model")
    st.write("Analisis performa model dengan metrik evaluasi dan visualisasi tambahan.")

    # Pilih kolom target
    target_col = st.selectbox("Pilih kolom target (label):", data.columns)

    if st.button("Tampilkan Performa Model"):
        with st.spinner("Menghitung metrik performa..."):
            data, label_encoders = preprocess_data(data, target_col)
            X = data.drop(columns=[target_col])
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Latih model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            # Tampilkan Precision, Recall, F1-Score
            st.write("### Precision, Recall, F1-Score:")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Tampilkan Confusion Matrix
            st.write("### Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            # Tampilkan AUC-ROC Curve (jika memungkinkan)
            # Tambahkan ROC Curve jika memungkinkan
            if len(model.classes_) == 2:  # Untuk kasus binary
                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import LabelEncoder

                # Encode target (y_test) menjadi numerik
                le = LabelEncoder()
                y_test_encoded = le.fit_transform(y_test)

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test_encoded, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)

                st.write("### AUC-ROC Curve:")
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.info("AUC-ROC Curve hanya tersedia untuk klasifikasi binary.")

# --- Feature Importance Page ---
elif menu == "Feature Importance":
    st.write("## Feature Importance")
    st.write("Analisis fitur mana yang paling berpengaruh dalam membuat prediksi menggunakan Random Forest.")

    # Pilih kolom target
    target_col = st.selectbox("Pilih kolom target (label):", data.columns)

    if st.button("Tampilkan Feature Importance"):
        with st.spinner("Menghitung feature importance..."):
            data, label_encoders = preprocess_data(data, target_col)
            X = data.drop(columns=[target_col])
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Latih model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Dapatkan feature importance
            feature_importance = model.feature_importances_
            feature_names = X.columns

            # Visualisasi feature importance
            st.write("### Peringkat Feature Importance:")
            importance_df = pd.DataFrame({
                "Fitur": feature_names,
                "Importance": feature_importance
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(importance_df)

            st.write("### Visualisasi Feature Importance:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Fitur", data=importance_df, palette="viridis", ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Fitur")
            st.pyplot(fig)

# --- Pemilihan Model Page ---
elif menu == "Coba Model Lain":
    st.write("## Pilih Model")
    st.write("Pilih algoritma machine learning yang ingin digunakan, latih model, dan evaluasi performanya.")

    # Pilihan algoritma
    model_options = ["Random Forest", "Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]
    selected_model = st.selectbox("Pilih model machine learning:", model_options)

    # Pilih kolom target
    target_col = st.selectbox("Pilih kolom target (label):", data.columns)

    if st.button("Latih Model"):
        with st.spinner(f"Melatih model {selected_model}, mohon tunggu..."):
            data, label_encoders = preprocess_data(data, target_col)
            X = data.drop(columns=[target_col])
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Inisialisasi model berdasarkan pilihan
            if selected_model == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif selected_model == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif selected_model == "K-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier()
            elif selected_model == "Support Vector Machine":
                from sklearn.svm import SVC
                model = SVC(probability=True, random_state=42)

            # Latih model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi performa model
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model {selected_model} berhasil dilatih dengan akurasi: {accuracy:.2f}")

            st.write("### Confusion Matrix:")
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix - {selected_model}")
            st.pyplot(fig)

            # Simpan model terbaik
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "feature_names": list(X.columns), "label_encoders": label_encoders}, f)
            st.success(f"Model {selected_model} disimpan ke '{model_path}'.")

# --- Halaman Help ---
elif menu == "Help":
    st.write("## Panduan Penggunaan Aplikasi")
    st.write("Selamat datang di halaman bantuan! Berikut adalah panduan untuk setiap fitur aplikasi:")

    # Penjelasan Home
    st.write("### 1. Home")
    st.write("""
    Halaman ini memberikan gambaran umum tentang dataset yang digunakan dalam aplikasi.
    - Anda dapat melihat statistik deskriptif dataset.
    - Tersedia juga berbagai visualisasi seperti histogram, scatterplot, dan heatmap korelasi.
    """)

    # Penjelasan Training Model
    st.write("### 2. Training Model")
    st.write("""
    Pada halaman ini, Anda dapat melatih model machine learning dengan dataset yang digunakan.
    - Pilih kolom target (label) yang ingin diprediksi.
    - Klik tombol 'Mulai Training' untuk melatih model dengan Random Forest.
    """)

    # Penjelasan Prediksi Data Baru
    st.write("### 3. Prediksi Obat")
    st.write("""
    Halaman ini memungkinkan Anda melakukan prediksi berdasarkan input baru.
    - Isi semua parameter (Umur, Rasio Na/K, Jenis Kelamin, dll.).
    - Klik tombol 'Prediksi' untuk mendapatkan hasil prediksi.
    """)

    # Penjelasan Riwayat Prediksi
    st.write("### 4. Riwayat Prediksi")
    st.write("""
    Halaman ini menampilkan semua prediksi yang telah dilakukan.
    - Anda dapat melihat data input dan hasil prediksi sebelumnya.
    - Tersedia opsi untuk mengunduh riwayat prediksi dalam format CSV.
    """)

    # Penjelasan Optimisasi Model
    st.write("### 5. Optimisasi Model")
    st.write("""
    Halaman ini memungkinkan Anda menyesuaikan hyperparameter model Random Forest.
    - Atur jumlah pohon (n_estimators) dan kedalaman maksimum (max_depth).
    - Latih ulang model dan lihat performanya.
    """)

    # Penjelasan Performa Model
    st.write("### 6. Performa Model")
    st.write("""
    Halaman ini menampilkan evaluasi performa model yang telah dilatih.
    - Tersedia metrik seperti akurasi, precision, recall, dan F1-score.
    - Visualisasi seperti Confusion Matrix dan AUC-ROC Curve juga ditampilkan.
    """)

    # Penjelasan Feature Importance
    st.write("### 7. Feature Importance")
    st.write("""
    Halaman ini menunjukkan fitur mana yang paling berpengaruh dalam membuat prediksi.
    - Tabel dan grafik feature importance ditampilkan untuk memudahkan interpretasi.
    """)

    # Penjelasan Pemilihan Model
    st.write("### 8. Pemilihan Model")
    st.write("""
    Halaman ini memberikan fleksibilitas dalam memilih algoritma machine learning.
    - Pilih model seperti Random Forest, Logistic Regression, KNN, atau SVM.
    - Bandingkan performa model untuk menentukan yang terbaik.
    """)

    # Penjelasan Mode Tampilan
    st.write("### 9. Mode Tampilan")
    st.write("""
    Anda dapat memilih mode tampilan aplikasi:
    - **Light Mode**: Tampilan terang (default).
    - **Dark Mode**: Tampilan gelap untuk kenyamanan mata.
    """)

    # Tambahan Informasi
    st.write("### Tambahan Informasi")
    st.write("""
    Jika Anda mengalami kendala dalam penggunaan aplikasi, hubungi pengembang untuk bantuan lebih lanjut.
    """)

st.markdown(
    """
    <div class="footer">
        © 2025 Aplikasi Klasifikasi Data Pasien | Dibuat oleh Fadhaeil Rizki Mahwendi|211220095
    </div>
    """,
    unsafe_allow_html=True,
)

