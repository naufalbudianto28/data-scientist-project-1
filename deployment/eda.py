import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def run():
    st.title('Eksplorasi Data pada Dataset Telemarketing Deposito Bank Portugis')

    data = pd.read_csv("/Users/Naufal's/Desktop/Hacktiv8 Tugas/p1-ftds031-rmt-m2-naufalbudianto28/bank-full.csv", sep=';')
    st.dataframe(data)

    st.subheader('Basic Data Distribution')

    st.write('#### 1. Age Distribution')
    fig, ax = plt.subplots(figsize=(18, 6))
    age_bar = data['age'].value_counts().sort_index().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(data['age'].unique())))
    ax.set_xticklabels(data['age'].unique(), rotation=45)

    for p in age_bar.patches:
        age_bar.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Dari hasil visualisasi data di atas, dapat disimpulkan bahwa mayoritas klien Bank Portugis didominasi oleh Early Adults (usia 25-35 tahun), dimana merupakan masa-masa produktif mayoritas orang. Mid-Adults (usia 36-45 tahun), dimana mayoritas orang sudah berkeluarga dan memiliki posisi mid-level pada karir. Dan Late-Adults (usia 46-60 tahun), dimana mayoritas orang mengalami masa puncak karir dan sudah bersiap untuk memasuki masa pensiun.')

    ###
    st.write('#### 2. Job Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    # Melakukan ploting bar terhadap distribusi jenis pekerjaan dan diurutkan dari nilai terbesar. 
    group_job = data['job'].value_counts().sort_values(ascending=False)
    group_job.plot(kind='bar', ax=ax)
    ax.set_title('Job Distribution')
    ax.set_xlabel('Job')
    ax.set_ylabel('Distribution')
    # Menampilkan nilai tiap bar dengan fungsi looping.
    for bar in ax.patches:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Selaras dengan hasil analisa visual pada distribusi umur klien, pada dataset klien Bank Portugis menunjukkan bahwa mayoritas klien mereka adalah pekerja (Adults - usia produktif). Namun jika diperhatikan dari visualisasi di atas, ada beberapa kejanggalan yaitu adanya typo pada tipe pekerjaan dengan value `admin.` yang seharusnya adalah `admin`. Terdapat tipe pekerjaan `unknown` yang dimana seharusnya **sudah terisi** untuk keperluan administrasi Bank Portugis. Maka beberapa kejanggalan di atas akan saya tangani saat proses Feature Engineering.')

    ###
    st.write('#### 3. Marital Status Distribution')
    marital_counts = data['marital'].value_counts()
    # Melakukan ploting pie-chart pada distribusi marital status.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(marital_counts, labels=marital_counts.index, autopct='%1.2f%%', startangle=140)
    ax.set_title('Marital Status')
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Hasil analisa visual pada status pernikahan menunjukkan bahwa klien Bank Portugis didominasi oleh orang yang sudah berkeluarga. Selaras dengan analisa pada distribusi umur dan tipe pekerjaan, dimana didominasi oleh umur 30-40 tahun yang di mayoritas masyarakat pada umumnya sudah berkeluarga, terlihat dari hasil analisa status pernikahan didominasi oleh **married** dengan `60,2%` kemudian **single** dengan jumlah `28,3%`.')

    ###
    st.write('#### 4. Education Background Distribution')
    ed_counts = data['education'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(ed_counts, labels=ed_counts.index, autopct='%1.2f%%', startangle=140)
    ax.set_title('Educational Background')
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Data educational background memiliki value primary (SD), secondary (SMP-SMA/SMK), dan tertiary (Kuliah). Dimana dari hasil eksplorasi visual didapatkan klien Bank Portugis didominasi oleh secondary (51,3%) dan tertiary (29,4%) educational background. Selaras dengan hasil visualisasi tipe pekerjaan, bahwa klien didominasi oleh blue-collar (merupakan buruh pekerja industri manual) yang mayoritas adalah lulusan SMK (secondary).')
    st.write('Namun hasil eksplorasi ini juga menunjukkan **kejanggalan**, yaitu adanya value `unknown` pada data educational background. Seharusnya dikarenakan ini termasuk biodata pribadi dan merupakan syarat administrasi, secara logika tidak ada educational background bernilai unknown pada dataset. Maka nantinya baris-baris yang memiliki value `unknown` pada educational background akan saya hapus disaat melakukan proses feature engineering.')

    ###
    st.write('#### 4. Balance Distribution')
    bins = [-float('inf'), 0, 25000, 50000, float('inf')]
    labels = ['Minus', '0 - 25,000', '25,000 - 50,000', '>= 50,000']

    data['balance_category'] = pd.cut(data['balance'], bins=bins, labels=labels, right=False)
    balance_distribution = data['balance_category'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    balance_distribution.plot(kind='bar', color='green', width=0.8, ax=ax)
    for i, v in enumerate(balance_distribution):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=10)

    ax.set_title('Balance Distribution')
    ax.set_xlabel('Range of Balance (€)')
    ax.set_ylabel('Count')
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Hasil analisa visual pada distribusi rata-rata saldo para klien Bank Portugis menunjukkan mayoritas klien memiliki rata-rata saldo di bank dengan range €0-25,000 dengan presentase 91,5%.')
    st.write('Dan ditemukan insight menarik yaitu visualisasi menunjukkan bahwa ada 8,3% klien yang memiliki saldo minus (< €0). Hal ini bisa disebabkan oleh beberapa faktor, seperti kemungkinan **gagal bayar** atau kredit macet, mengingat Bank Portugis juga menyediakan layanan kartu kredit, pinjaman personal, dan cicilan rumah. Dan atau **tidak ada pemasukan** lalu terpotong biaya administrasi bulanan, jika dilihat dari hasil visualisasi pekerjaan ada ~8% klien yang tidak bekerja dan sudah pensiun.')

    ###
    st.write('#### 4. Credit and Loan Distribution')
    default_counts = data['default'].value_counts()
    housing_counts = data['housing'].value_counts()
    loan_counts = data['loan'].value_counts()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.pie(default_counts, labels=default_counts.index, autopct='%2.2f%%', startangle=140)
    ax1.set_title('Have Credit?')
    ax2.pie(housing_counts, labels=housing_counts.index, autopct='%2.2f%%', startangle=140)
    ax2.set_title('Have Housing Loan?')
    ax3.pie(loan_counts, labels=loan_counts.index, autopct='%2.2f%%', startangle=140)
    ax3.set_title('Have Personal Loan?')
    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Grafik pie di atas menunjukkan presentase penggunaan layanan peminjaman/cicilan yang dimiliki Bank Portugis oleh para kliennya. Dimana layanan cicilan rumah paling diminati oleh para klien Bank Portugis dengan presentase 55,6% pengguna. Secara logis hal ini sesuai dengan demografi klien Bank Portugis yang mayoritas adalah perkerja dan sudah berkeluarga, dimana pasti ada kebutuhan untuk memiliki tempat tinggal yang layak bagi keluarga mereka.')

    ###
    st.subheader('Telemarketing Data Distribution')
    st.write('#### 1. Telemarketing Type Distribution')
    
    tc_counts = data['contact'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(tc_counts, labels=tc_counts.index, autopct='%2.2f%%', startangle=140)
    ax.set_title('Type of Telemarketing Contact')
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Visualisasi di atas menunjukkan tentang tipe usaha telemarketing yang Tim Telemarketing Bank Portugis lakukan untuk melakukan tawaran campaign-campaign yang mereka miliki, jika dilihat bahwa klien mereka cenderung lebih aktif / responsif melalui telfon seluler (HP) dengan nilai 65% dibandingkan dengan telfon rumah yang bernilai 6,4%. Namun juga terdapat tipe telemarketing unknown, hal ini bisa disebabkan oleh data pada klien yang belum pernah dihubungi atau dilakukan campaign, sehingga bernilai unknown. Pada data ini nantinya tidak akan saya tangani terkait value unknown pada kolom contact, dikarenakan untuk menunjukkan keaslian data yang berpengaruh kepada insight Tim Telemarketing.')

    ###
    st.write('#### 2. Duration of The Contact Distribution')
    bins = [-float('inf'), 60, 900, 1800, 2700, 3600, float('inf')]
    labels = ['0 - 1 minute', '1 - 15 minutes', '15 - 30 minutes', 
            '30 - 45 minutes', '45 - 60 minutes', '>= 60 minutes']
   
    data['dur_category'] = pd.cut(data['duration'], bins=bins, labels=labels, right=False)
    dur_distribution = data['dur_category'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    dur_distribution.plot(kind='bar', color='purple', width=0.8, ax=ax)

    for i, v in enumerate(dur_distribution):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontsize=10)

    ax.set_title('Call Duration Distribution')
    ax.set_xlabel('Call Duration')
    ax.set_ylabel('Count')
    ax.set_xticklabels(labels, rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Terlihat dari hasil eksplorasi visual terhadap kolom durasi waktu telfon terakhir yang dilakukan Tim Telemarketing Bank Portugis, bahwa mayoritas telfon yang mereka lakukan dengan klien adalah kurang dari 30 menit. Hal ini menunjukkan bahwa mayoritas dari klien mereka adalah orang yang sibuk (tidak punya banyak waktu), mengingat hal ini juga ditunjukkan dari hasil eksplorasi pekerjaan dan umur, bahwa klien mereka merupakan usia produktif bekerja. Namun ada hal menarik pada grafik bahwa terdapat sekitar 4659 telfon dengan durasi 0 hingga 1 menit, hal ini menunjukkan bahwa ada beberapa telfon yang belum dilakukan oleh Tim kepada klien, dan atau penolakan telfon.')

    ###
    st.write('#### 3. Number of Campaign Calls Distribution')
    call_distribution = data['age'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(18, 6))
    call_bar = call_distribution.plot(kind='bar', color='magenta', ax=ax)

    ax.set_title('Number of Calls Distribution')
    ax.set_xlabel('Number of Calls')
    ax.set_ylabel('Count')
    ax.set_xticklabels(call_distribution.index, rotation=45)

    for c in call_bar.patches:
        ax.annotate(str(c.get_height()), (c.get_x() + c.get_width() / 2., c.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    st.pyplot(fig)
    st.write('Visualisasi di atas merupakan distribusi jumlah telfon yang dilakukan Tim Telemarketing terhadap para klien dalam kurun waktu 1 tahun. Hasil grafik bisa jadi menunjukkan terhubung atau tidak terhubungnya suatu usaha telfon yang dilakukan Tim Telemarketing mereka.')
    st.write('Dan ada juga kemungkinan tiap individu Tim Telemarketing membatasi maksimum usaha menelfon nya. Contoh; si X membatasi maksimum jumlah call ke klien Y sebanyak 30x telfon dalam satu tahun, sehingga ketika sudah mencapai batas tersebut dan tidak ada jawaban, maka X memberikan status unknown pada CRM mereka, hal ini lazim terjadi pada Tim Sales di semua industri.')

    ###
    st.write('#### 4. Previous Campaign Result')
    pc_counts = data['poutcome'].value_counts()

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.pie(pc_counts, labels=pc_counts.index, autopct='%2.2f%%', startangle=140)
    ax.set_title('Previous Campaign Result')
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)
    st.write('Pie Chart di atas menunjukkan hasil dari campaign sebelumnya tentang layanan/produk berbeda, dengan menghasilkan output mayoritas bernilai unknown, dimana ada kemungkinan hal ini terjadi karena Telfon tidak di respon oleh klien, Nomor tidak bisa dihubungi oleh Tim Telemarketing, Telemarketing terblokir oleh klien, dan atau Klien belum dihubungi oleh Telemarketing. Sehingga mereka melakukan update status pada CRM mereka dengan nilai `unknown`')

    ###
    st.subheader('Correlation between Feature and Target')
    st.write('#### 1. Numerical Columns')
    st.write('Pada tahap ini saya melihat heatmap korelasi Spearman (dengan asumsi masih terdapat outliers) terhadap kolom-kolom numerikal dan kolom target.')
    # Membuat dataset dummy.
    df_dummy = pd.get_dummies(data, columns=['y'], drop_first=True, dtype=int)
    df_num = df_dummy[['age', 'balance', 'duration', 'campaign', 'day', 'pdays', 'previous', 'y_yes']]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_num.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Numerical - Spearman Correlation")
    st.pyplot(fig)
    st.write('Untuk korelasi Spearman di atas saya menentukan threshold korelasi adalah corr >= |0,05|, untuk dikategorikan berkorelasi dengan target y_yes, sehingga fitur-fitur yang memiliki korelasi adalah: balance, duration, campaign, pdays, previous.')

    ###
    st.write('#### 2. Categorical Columns')
    st.write('Pada tahap ini saya melihat heatmap korelasi Kendall terhadap kolom kategorikal dan kolom target.')

    df_cat = df_dummy[['job', 'marital', 'contact', 'month', 'poutcome', 'education', 'default', 'housing', 'loan']]
    cols_1 = []
    corr_1 = []
    pval_1 = []

    for col in df_cat.columns:
        corr_coef, p_value = kendalltau(df_cat[col], df_dummy['y_yes'])
        cols_1.append(col)
        corr_1.append(corr_coef)
        pval_1.append(p_value)

    corr_df = pd.DataFrame({'Nama Kolom': cols_1, 'Kendall Corr': corr_1, 'P-value': pval_1})
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_df.set_index('Nama Kolom')[['Kendall Corr']].T, annot=True, cmap='coolwarm', cbar=True, ax=ax)
    ax.set_title('Kendall Correlation Heatmap Between Features and Target')
    st.pyplot(fig)
    st.write('Untuk korelasi Kendall di atas saya menentukan threshold adalah corr >= |0,035|, untuk dikategorikan berkorelasi dengan target, sehingga fitur-fitur yang memiliki korelasi adalah: job, marital, education, contact, housing, loan, dan poutcome.')


    ###
    st.subheader('Target Proportion')
    y_count = data['y'].value_counts()

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.pie(y_count, labels=y_count.index, autopct='%2.2f%%', startangle=140)
    ax.set_title('Target Proportion')
    ax.axis('equal')
    st.pyplot(fig)
    st.write('Terlihat hasil dari program campaign Tim Telemarketing Bank Portugis tentang layanan deposito berjangka yang telah dilakukan selama satu tahun, memiliki mayoritas penolakan dengan nilai yang sangat dominan 88%. Diharapkan insight-insight yang didapatkan dari proses eksplorasi data dapat menjadi acuan dan bahan evaluasi Tim Telemarketing kedepannya.')

if __name__ == '__main__':
    run() 