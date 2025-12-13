import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title='NLP Project - EDA & Insight', 
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

    if st.button('**Prediksi Sentimen**', width="stretch"):
        st.switch_page('pages/st_page4.py')


st.title('Exploratory Data Analysis & Insight')
st.divider()

#Menampilkan informasi dari data
st.title('1. Data Information (df.info)')
st.divider()

header = ['ID', 'Topic', 
'Sentiment', 'Tweet']
df = pd.read_csv(
    'https://raw.githubusercontent.com/comradeftj/Datsets/refs/heads/main/twitter_training.csv',
    names=header,
    header=None
)
df = df.head(10000)

dtypes_df = pd.DataFrame({
    "dtype": df.dtypes,
    "missing_values": df.isnull().sum(),
    "unique_values": df.nunique()
})
st.dataframe(dtypes_df)

st.markdown("""
        <p style='font-size:18px; text-align:justify;'> 
         Berdasarkan hasil dari df.info berikut, diperoleh insight bahwa
         dalam dataset ini memiliki 4 features dimana 'ID' memiliki tipe
         data int64 dan 'Topic', 'Sentiment', dan 'Tweet' memiliki
         tipe data object yang isi datanya berupa string.<br>
         Atribut 'ID', 'Sentiment', dan 'Tweet' tidak memiliki <i>missing value</i>, sedangkan
         'Tweet' memiliki 112 missing values yang nantinya akan ditangani lebih lanjut di bagian
         pre-processing data.<br>
         Setiap atribut memiliki nilai unik yang menyamakan suatu baris dengan baris lainnya, 
         'ID' memiliki nilai unik sejumlah 1667 data, 'Topic' memiliki 5 data unik, 'Sentiment'
         memiliki 4, dan puncaknya adalah 'Tweet' yang hampir seluruh datanya merupakan nilai unik
         sejumlah 9362 data.
         </p>
         """, unsafe_allow_html=True)

st.divider()

#Missing value
st.title("2. Missing Value Detection")
st.divider()

fig = px.bar(
    df.isnull().sum().reset_index(),
    x='index', y=0,
    labels={'index':'Kolom', 0:'Jumlah Missing'}
)
st.plotly_chart(fig)

st.markdown("""
        <p style='font-size:18px; text-align:justify;'> 
         Berdasarkan hasil deteksi missing value, diperoleh atribut 'Tweet'
        memiliki missing value sejumlah 112 data.
         </p>
         """, unsafe_allow_html=True)

st.divider()

#Categorical Data Distribution
st.title('3. Categorical Data Distribution')
st.divider()

#distribusi atribut 'topic'
st.write("""
         <p style='font-size:18px; text-align:center;'> 
         <b>Topic Feature Distribution<b></p>""",unsafe_allow_html=True)

all_topic = ["CallOfDutyBlackopsColdWar", "Overwatch", "Amazon", "Borderlands", "Xbox(Xseries)"]
with st.container(border=True):
    topics = st.multiselect("Topics", all_topic, default=all_topic)

freq_topic = df[df['Topic'].isin(topics)]['Topic'].value_counts().reset_index()
freq_topic.columns = ['Topic', 'Frequency']

tab1, tab2 = st.tabs(["Chart", "Dataframe"])
#bar chart
with tab1:
    fig = px.bar(
        freq_topic,
        x='Topic',
        y='Frequency',
        color='Topic',
        text='Frequency'
    )
    st.plotly_chart(fig)
    st.write("""
            <p style='font-size:18px; text-align:justify;'> 
            Berdasarkan hasil distribusi data kategorikal atribut 'Topic', 
            diperoleh kesimpulan bahwa topik yang paling banyak dibahas adalah
            CallOfDutyBlackopsColdWar sejumlah 2376 data diikuti dengan Overwatch, 
            Amazon, Borderlands, dan Xbox (Xservies) dimana kebanyakan dari tweet ini
            pembahas mengenai permainan video game. </p>
             """, unsafe_allow_html=True)
#tabel dataframe
with tab2:
    st.dataframe(freq_topic, width="stretch")

#distribusi atribut 'sentiment'
st.write("""
         <p style='font-size:18px; text-align:center;'> 
         <b>Sentiment Feature Distribution<b></p>""",unsafe_allow_html=True)

all_sent = ["Positive", "Neutral", "Negative", "Irrelevant"]
with st.container(border=True):
    sentiment = st.multiselect("Sentiment", all_sent, default=all_sent)

freq_sent = df[df['Sentiment'].isin(sentiment)]['Sentiment'].value_counts().reset_index()
freq_sent.columns = ['Sentiment', 'Frequency']

tab1, tab2 = st.tabs(["Chart", "Dataframe"])
#bar chart
with tab1:
    fig = px.bar(
        freq_sent,
        x='Sentiment',
        y='Frequency',
        color='Sentiment',
        text='Frequency'
    )
    st.plotly_chart(fig)
    
    st.write("""
            <p style='font-size:18px; text-align:justify;'> 
            Berdasarkan hasil distribusi data kategorikal atribut 'Sentiment', 
            diperoleh kesimpulan bahwa jenis sentiment didominasi dengan sentiment positive
            diikuti dengan sentimen netral, negatif, dan terdapat juga tweet yang
            tidak berhubungan dengan topik (irrelevant)
             </p>""", unsafe_allow_html=True)
    
#tabel dataframe
with tab2:
    st.dataframe(freq_sent, width="stretch")

st.divider()

# Text count distribution
st.title("4. Text Word Count Distribution")
st.divider()

def word_count(text):
    if pd.isna(text):
        return 0
    text = text.strip(' ').split(' ')
    return len(text)

def char_count(text):
    if pd.isna(text):
        return 0
    text = text.strip(' ').replace(' ', '')
    return len(text)

df['word_count'] = df['Tweet'].apply(word_count)
df['char_count'] = df['Tweet'].apply(char_count)
cols = ['word_count', 'char_count']

tab1, tab2 = st.tabs(["Histogram + KDE", "Boxplot"])

#Histogram dan KDE
with tab1:
    st.header("Histogram & KDE per Kolom")
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
                <p style='font-size:16px; text-align:justify;'>
                 Berdasarkan hasil histogram word count distribution, 
                 diperoleh distribusi right-skew yang artinya kebanyakan 
                 tweet memiliki jumlah kata yang sedikit yang puncaknya di 5-20 kata </p>
                """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
                <p style='font-size:16px; text-align:justify;'>
                 Berdasarksn hasil histogram char count distribution, 
                 diperoleh distribusi right-skew dimana hal ini berkorelasi
                 dengan word count, semakin banyak kata yang terhitung, maka semakin
                 banyak karakter yang terdeteksi.</p>
                """, unsafe_allow_html=True)
#Boxplot
with tab2:
    st.header("Outliers per Kolom (Boxplot)")
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    for i, col in enumerate(cols):
        sns.boxplot(x=df[col], ax=axes[i], color='lightgreen')
        axes[i].set_title(f'{col} Outliers', fontsize=14, fontweight='bold')
    st.pyplot(fig)

    st.markdown("""
                 <p style='font-size:16px; text-align:justify;'>
                 Outlier pada word dan char count disebabkan karena lebih banyak
                orang yang menulis tweet dengan jumlah kata yang sedikit, dibandingkan 
                dengan kata yang panjang. Sehingga menyebabkan data dengan kalimat panjang ini menjadi data outlier. 
                Namun, hal ini masih tergolong normal, karena dalam menulis tweet di media sosial tidak ada batasan minimal
                ataupun maksimal dari penulisan.</p>
                """, unsafe_allow_html=True)