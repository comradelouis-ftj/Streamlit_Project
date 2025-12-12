import streamlit as st

from st_functions import load_image

st.set_page_config(
    page_title='NLP Project - Preprocessing', 
    layout='wide'
)

st.markdown(
"""
<style>
div[data-testid="stSidebarNav"] {display: none;}
.stButton > button {
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>
""", 
unsafe_allow_html=True)

with st.sidebar:
    st.title('NLP Dengan Neural Network')
    st.divider()

    if st.button('**Profil Proyek**', width="stretch"):
        st.switch_page('st_page1.py')
    
    if st.button('**EDA/Insight**', width="stretch"):
        st.switch_page('pages/st_page2.py')
    
    if st.button('**Tahap Preprocessing**', width="stretch"):
        st.switch_page('pages/st_page3.py')

    if st.button('**Try the Model!**', width="stretch"):
        st.switch_page('pages/st_page4.py')


st.title('Tahapan Preprocessing')
st.text("""
    Preprocessing data adalah tahapan yang dilakukan untuk mempersiapkan data sebelum masuk ke tahap analisis dan visualisasi. 
        
    Dengan adanya preprocessing data, maka dapat mengurangi data noise, drop data yang kurang relevan dengan analisis, melakukan proses sehingga data bisa dianalisis dengan optimal.Preprocessing data dapat dilakukan dengan membersihkan data, transformasi data, encoding, dll.
            
    Berikut ini adalah tahapn tahapan preprocessing yang akan di lakukan  """) 

st.subheader("1. Penanganan data NaN atau nilai yang kosong")
st.divider()
st.markdown("#####  Penanganan Data Irrelevant")

del_sentimen = load_image("(1)apus_sentimen.png")
st.image(del_sentimen,caption="Mengganti data irrelevant dengan netral",width=400)
st.text("""Karena analisis sentimen ini hanya difokuskan pada positif, negatif, dan netral maka data yang irrelevant akan dimasukkan ke kategori netral.
Ini karena sentimen yang irrelevant berarti tidak berhubungan dengan sentimen positif maupun negatif, sehingga masih valid bila dimasukkan ke kategori netral.""")
dropna = load_image("(2)dropna.png")
st.markdown("#####  Menghapus Data kosong atau NaN")
st.image(dropna,caption="Menghapus data dengan nilai kosong",width=400)
st.text("""Menghapus data NaN atau nilai yang kosong. Maka dari itu, kolom ‘Tweet’ yang memiliki 112 nilai NaN akan di drop karena tidak terlalu berpengaruh pada data yang besar sejumlah 10000 data.""")

st.subheader("2. Data encoding")
st.divider()
st.markdown("#####  Mengubah data kategori menjadi bentuk numerik")
encoding = load_image("(3)encoding.png")
st.image(encoding,caption="Encoding ‘Sentiment’",width=400)
st.text("Data encoding merupakan tahapan untuk mengubah data kategorik menjadi format numerik sehingga memudahkan dalam analisis terutama disebabkan karena adanya data yang berulang. Jenis encoding yang digunakan adalah Label encoding karena nilai dari variabel lebih dari 2 sehingga tidak bisa menggunakan one hot encoding (encoding biner). Maka dari itu, kami mengubah Sentimen Positif menjadi angka 2, Sentimen Negatif menjadi angka 0, dan Sentimen Netral menjadi angka 1. ")

st.subheader("3. Menghapus tanda baca, karakter, dan emoji ")
st.divider()
st.markdown("##### Removing Punctuations & Emoji Handling")
apus_punc = load_image("(4)apuspunctuation.png")
head_punc = load_image("(5)headpunctuation.png")
st.image(apus_punc, caption="Removing Punctuations & Emoji Handling",width=400)
st.image(head_punc, caption="Hasil",width=350)
st.text("Selanjutnya, data teks dibersihkan dengan menghapus tanda baca, karakter, dan emoji sehingga analisis hanya berfokus pada perkataan dalam teks yang tidak dipengaruhi oleh tanda baca apapun sehingga meningkatkan akurasi dan lebih memudahkan ML untuk memahami sentimen dari tweet tersebut.")

st.subheader("4. Tokenizing data")
st.divider()
st.markdown("##### *Tokenizing data*")
tokenize = load_image("(6)tokenizing.png")
st.image(tokenize, caption="*Tokenizing*",width=400)
st.text("Tokenizing data merupakan proses untuk memecahkan teks menjadi token-token agar lebih spesifik dan lebih mudah dipahami oleh ML. Jenis tokenisasi yang kamu gunakan adalah tokenisasi kata sehingga hanya memisahkan setiap kata dalam kalimat. Contohnya adalah ‘im getting on borderlands and will kill you all,’ maka akan dipisah menjadi [im, getting, on, borderlands, and, will, kill, you, all]. Maka, kata yang sudah terpisah ini dapat memudahkan ML untuk melakukan analisis dan pengelompokkan data sesuai dengan jenisnya (Lemmatization).")

st.subheader("5. Stopword Removal")
st.divider()
st.markdown("##### *Stopword Removal*")
stopword = load_image("(7)stopword.png")
stopword_head = load_image("(8)stopwordhead.png")
st.image(stopword, caption="*Stopword Removal*",width=400)
st.image(stopword_head, caption="Hasil",width=400)
st.text("""
Selanjutnya Stopword Removal akan dilakukan terhadap data, Stopword Removal merupakan bagian dari tahapan preprocessing teks yang bertujuan untuk menghapus kata yang tidak relevan di dalam suatu kalimat berdasarkan daftar stopword.
 
Daftar stopword yang biasa digunakan berbentuk digital library yang daftarnya sudah tersedia sebelumnya, namun tidak semua kata-kata yang terdapat didalam digital library merupakan kata yang tidak relevan dalam suatu data tertentu.
        
Dalam model ini stopwords yang akan di hilangkan akan berasal dari dari modul corpus yang ada di dalam Library Natural Language Toolkit (NLTK). Sehingga yang akan di hapus adalah kata seperti : `the`, `and`, `is` and `in`. Sehingga kalimat yang tersedia di dalam data akan lebih sedikit seperti gambar output di atas.
""")


st.subheader("6. Lemmatization & POS Tagging")
st.divider()
st.markdown("##### *Lemmatization & POS Tagging*")
lemmapos = load_image("(9)lemma&pos.png")
st.image(lemmapos, caption="*Lemmatization & POS Tagging*",width=400)
st.text(""" Selanjutnya akan dilakukan proses Lematisasi & Pos Tagging, Lemmatization/lematisasi adalah proses mengelompokkan berbagai bentuk kata-kata dari suatu kata berimbuhan menjadi satu bentuk dasarnya, yang dikenal sebagai lemma atau bentuk kamus. Proses ini melibatkan analisis linguistik dan konteks untuk memastikan bentuk dasar yang dihasilkan adalah kata yang valid. POS tagging (Part-of-Speech tagging) dalam NLP adalah proses mengategorikan setiap kata dalam sebuah kalimat ke dalam kelas katanya (seperti kata benda, kata kerja, kata sifat, dll.). 

Dalam model ini pertama tama konversi tag POS (Part of Speech) hasil dari nltk.pos_tag() menjadi format yang bisa digunakan oleh WordNetLemmatizer. Lalu semua data yang telah diproses akan dilakukan proses pos tagging yaitu mengelompokkan kata sesuai dengan kategorinya masing-masing, selanjutnya akan dilakukan proses lematisasi dengan mengubah kata ke bentuk dasar, dan pasangkan kata dan kategori/tag menjadi : (kata, tag) untuk setiap kata dalam input.    """)

st.subheader("7. Data Splitting")
st.divider()
st.markdown("##### *Data Splitting*")
datasplit = load_image("(10)dataspliting.png")
st.image(datasplit,caption="*Data Splitting*",width=600)

st.text("Selanjutkan kita pisahkan data menjadi 2 untuk memisahkan data training (80%) dan data testing (20%) untuk melatih Machine Learning.")


st.subheader("8. Vectorizing & Undersampling")
st.divider()
st.markdown("##### *Vectorizing & Undersampling*")
vectorunder = load_image("(11)vector&under.png")
head_vu = load_image("(12)headvector&under.png")
st.image(vectorunder,caption="*Vectorizing & Undersampling*",width=400)
st.image(head_vu,caption="Hasil",width=400)

st.text("""
        Selanjutnya akan dilakukan proses vectorizing & undersampling, agar data dapat diproses dengan model untuk menghasilkan output yang sesuai diperlukannya vektorisasi terhadap data, Vektorisasi adalah proses mengubah data dari format mentah ke dalam representasi numerik yang disebut vektor.Dalam model ini hasil program akan menghasilkan output numerik 0,1,2 dari hasil perhitungan berdasarkan sentiment dan tweet.

        Sebelum melakukan vektorisasi data akan di undersampling terlebih dahulu , Undersampling adalah salah satu teknik untuk mengatasi masalah ketidakseimbangan kelas (imbalanced dataset), di mana jumlah sampel dari satu kelas (kelas mayoritas) jauh lebih banyak daripada kelas lainnya (kelas minoritas).
        
        Dalam model kali ini dilakukan undersampling pada kelas sentiment yang memiliki nilai positif karena kelas positif memiliki nilai yang jauh lebih banyak dengan kelas lainnya sehingga diperlukannya undersampling agar model akan belajar dengan lebih dan akurat karena data yang di proses imbang sehingga tidak terlalu banyak belajar di satu kelas saja.
""")