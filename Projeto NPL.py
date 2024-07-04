#!/usr/bin/env python
# coding: utf-8

# ## Classificação e exploração

# In[1]:


# prompt: conecte o drive

from google.colab import drive
drive.mount('/content/drive/')


# In[6]:


import pandas as pd

resenha = pd.read_csv("/content/drive/MyDrive/imdb-reviews-pt-br.csv")
resenha.head()


# In[3]:


print("Negativa \n")
print(resenha.text_pt[189])


# In[4]:


print("Positivo \n")
print(resenha.text_pt[49002])


# In[5]:


print(resenha.sentiment.value_counts())


# In[7]:


classificacao = resenha["sentiment"].replace(["neg", "pos"], [0,1])

resenha["classificacao"] = classificacao


# In[7]:


resenha.head()


# ## Bag of Words: criando representações da linguagem humana

# In[8]:


from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform(resenha.text_pt)
print(bag_of_words.shape)


# In[9]:


matriz_densa = pd.DataFrame(bag_of_words.toarray(),
                            columns=vetorizar.get_feature_names_out())

matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words,
                      columns=vetorizar.get_feature_names_out())

matriz_esparsa.head()


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[11]:


def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                              texto[coluna_classificacao],
                                                              test_size = 0.2,
                                                              random_state = 42)
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)
    return regressao_logistica.score(teste, classe_teste)
print(classificar_texto(resenha, "text_pt", "classificacao")) # primeira acurácia


# ## Visualizando os dados com WordCloud.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud

todas_palavras = ' '.join([texto for texto in resenha.text_pt])

nuvem_palvras = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(todas_palavras)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.imshow(nuvem_palvras, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:


def nuvem_palavras_neg(texto, coluna_texto):
    texto_negativo = texto.query("sentiment == 'neg'")
    todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])

    nuvem_palvras = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(todas_palavras)
    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palvras, interpolation='bilinear')
    plt.axis("off")
    plt.show()

nuvem_palavras_neg(resenha, "text_pt")


# In[ ]:


def nuvem_palavras_pos(texto, coluna_texto):
    texto_positivo = texto.query("sentiment == 'pos'")
    todas_palavras = ' '.join([texto for texto in texto_positivo[coluna_texto]])

    nuvem_palvras = WordCloud(width= 800, height= 500,
                              max_font_size = 110,
                              collocations = False).generate(todas_palavras)
    plt.figure(figsize=(10,7))
    plt.imshow(nuvem_palvras, interpolation='bilinear')
    plt.axis("off")
    plt.show()

nuvem_palavras_pos(resenha, "text_pt")


# ## Tokenização e NLTK

# In[12]:


import nltk
from nltk import tokenize

token_espaco = tokenize.WhitespaceTokenizer()
todas_palavras = ' '.join([texto for texto in resenha.text_pt])
token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                   "Frequência": list(frequencia.values())})

df_frequencia.head()


# ## Visualização da Frequência

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


def pareto(texto, coluna_texto, n):
  todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])
  token_frase = token_espaco.tokenize(todas_palavras)
  frequencia = nltk.FreqDist(token_frase)
  df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),
                                   "Frequência": list(frequencia.values())})
  df_frequencia = df_frequencia.nlargest(columns="Frequência", n = n)
  plt.figure(figsize=(10,8))
  ax = sns.barplot(data = df_frequencia, x = "Palavra", y = "Frequência", color = "gray")
  ax.set(ylabel = "Contagem")
  plt.show()

pareto(resenha, "text_pt", 10)


# In[ ]:





# ## Aplicação das Stopwords

# Removendo stopwords padrão

# In[15]:


# Removendo stopwords

import nltk
nltk.download('stopwords')

palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")

frase_processada = list()
for opiniao in resenha.text_pt:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra) # juntando as palavras em frases
    frase_processada.append(' '.join(nova_frase)) # juntando todas as frases

resenha["tratamento_1"] = frase_processada # frases sem stopwords


# In[16]:


acuracia_teste = classificar_texto(resenha, "tratamento_1", "classificacao")
print(acuracia_teste) # segunda acurácia


# Agregando pontuação às stopwords pra remoção

# In[17]:


# Adicionado a pontuação às stopwords

from string import punctuation
token_pontuacao = tokenize.WordPunctTokenizer()

pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)

pontuacao_stopwords = pontuacao + palavras_irrelevantes


# In[18]:


# Removendo pontuação

frase_processada = list()
for opiniao in resenha["tratamento_1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra) # juntando as palavras em frases
    frase_processada.append(' '.join(nova_frase)) # juntando todas as frases

resenha["tratamento_2"] = frase_processada


# In[19]:


acuracia_tratamento_2 = classificar_texto(resenha, "tratamento_2", "classificacao")
print(acuracia_tratamento_2) # terceira acurácia


# Removendo acentos

# In[20]:


get_ipython().system('pip install unidecode')
import unidecode


# In[21]:


# Removendo acento das stopwords

stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]


# In[22]:


sem_acentos = [unidecode.unidecode(texto) for texto in resenha["tratamento_2"]] # lista tratamento_2 sem acento
resenha["tratamento_3"] = sem_acentos # coluna tratamento_2 sem acento

frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra) # juntando as palavras em frases
    frase_processada.append(' '.join(nova_frase)) # juntando todas as frases

resenha["tratamento_3"] = frase_processada


# In[23]:


acuracia_tratamento_3 = classificar_texto(resenha, "tratamento_3", "classificacao")
print(acuracia_tratamento_3) # quarta acurácia


# Padronizando texto em minúsculo

# In[24]:


frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_4"] = frase_processada


# In[25]:


acuracia_tratamento_4 = classificar_texto(resenha, "tratamento_4", "classificacao")
print(acuracia_tratamento_4) # quinta acurácia (resultado inferior ao anterior)


# ## Stemmetização

# In[ ]:


import nltk
nltk.download('rslp')
stemmer = nltk.RSLPStemmer()

frase_processada = list()
for opiniao in resenha["tratamento_4"]:
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
      nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_5"] = frase_processada


# In[ ]:


acuracia_tratamento_5 = classificar_texto(resenha, "tratamento_5", "classificacao")
print(acuracia_tratamento_5) # sexta acurácia


# ## TFIDF

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

tfidf_bruto = tfidf.fit_transform(resenha["text_pt"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto,
                                                              resenha["classificacao"],
                                                              test_size = 0.2,
                                                              random_state = 42)
regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_bruto = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_bruto) # resultado usando TfidVectorizer


# In[12]:


print(acuracia_teste) # comparando com resultado usando CountVectorizer


# In[ ]:


# Acurácia com as 50 features

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

tfidf_tratados = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados,
                                                              resenha["classificacao"],
                                                              test_size = 0.2,
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_tratados = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_tratados)


# ## N-grams

# In[ ]:


from nltk import ngrams


# In[ ]:


# Acurácia com n-grams

tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                              resenha["classificacao"],
                                                              test_size = 0.2,
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_ngrams)


# In[ ]:


# Acurácia com base inteira

tfidf = TfidfVectorizer(lowercase=False)
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,
                                                              resenha["classificacao"],
                                                              test_size = 0.2,
                                                              random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf)


# In[ ]:


pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index = tfidf.get_feature_names()
)

pesos.nlargest(10,0)


# In[ ]:


pesos.nsmallest(10,0)

