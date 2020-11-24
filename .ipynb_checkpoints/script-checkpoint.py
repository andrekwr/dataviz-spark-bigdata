from pyspark import SparkContext
import math
import pandas as pd

#Quantas ocorrencias da palavra em todos documentos
def conta_palavras(item):
    texto = item[1]
    palavras = texto.strip().split()
    return [(palavra.lower(),1) for palavra in palavras]

#Conta quantos docs cada palavra aparece
def conta_docs(item):
    texto = item[1]
    palavras = texto.strip().split()
    return [(palavra.lower(),1) for palavra in set(palavras)]

#Filtra palavras que aparecem em certa quantidade de docs
def filtra(item):
    contagem = item[1]
    return (contagem < doc_freq_max) and (contagem > doc_freq_min)

#Calcula idf de cada palavra filtrada
def computa_idf(item):
    palavra, contagem = item
    idf = math.log10(total_docs / contagem)
    return (palavra, idf)

if __name__ == '__main__':
    sc = SparkContext(appName="Megadados")
    #rdd = sc.sequenceFile("s3://megadados-alunos/web_brasil_small")
    rdd = sc.sequenceFile("pages/part-00000")
    #total de documentos na base
    total_docs = rdd.count()
    
    rdd_docs_word = rdd.flatMap(conta_docs).reduceByKey(lambda x,y: x + y).cache()
    
    #RDD com quantos docs cada palavra aparece considerando intervalo limite de 5 a 0.7*total_documentos
    rdd_docs_filtrado = rdd_docs_word.filter(filtra)
    rdd_idf = rdd_docs_filtrado.map(computa_idf)    
    
    #Filtrar RDDs para selecionar conjuntos com cada palavra definida.
    rdd_oreo = rdd.filter(lambda x: "oreo" in x[1]) 
    rdd_negresco = rdd.filter(lambda x: "negresco" in x[1])
    
    #Cálculo da frequência para os RDDs dos dois conjuntos.
    rdd_freq_oreo = rdd_oreo.flatMap(conta_palavras).reduceByKey(lambda x,y: x + y).map(lambda x: (x[0], math.log10(1 + x[1]))).cache()
    rdd_freq_negresco = rdd_negresco.flatMap(conta_palavras).reduceByKey(lambda x,y: x + y).map(lambda x: (x[0], math.log10(1 + x[1]))).cache()
    
    #Intersecção dos dois conjuntos de frequência.
    rdd_inter = rdd_freq_oreo.intersection(rdd_freq_negresco)
    
    #Calcula 100 palavras mais relevantes em conjunto às palavras escolhidas
    rdd_all = rdd_inter.join(rdd_idf).map(lambda x: (x[0], x[1][0]*x[1][1])).takeOrdered(100, key=lambda x: -x[1])
    df = pd.DataFrame(rdd_all, columns = ["palavra", "relevancia"])
    df_csv = df.to_csv("rdd_all.csv", index=False)
    
    #Calcula 100 palavras mais relevantes com apenas do conjunto Oreo
    rdd_oreo = rdd_freq_oreo.subtractByKey(rdd_freq_negresco).join(rdd_idf).map(lambda x: (x[0], x[1][0]*x[1][1])).takeOrdered(100, key=lambda x: -x[1])
    df_o = pd.DataFrame(rdd_oreo, columns = ["palavra", "relevancia"])
    df_o_csv = df_o.to_csv("rdd_oreo.csv", index=False)
    
    #Calcula 100 palavras mais relevantes com apenas do conjunto Negresco
    rdd_negresco = rdd_freq_negresco.subtractByKey(rdd_freq_oreo).join(rdd_idf).map(lambda x: (x[0], x[1][0]*x[1][1])).takeOrdered(100, key=lambda x: -x[1])
    df_n = pd.DataFrame(rdd_negresco, columns = ["palavra", "relevancia"])
    df_n_csv = df_n.to_csv("rdd_negresco.csv", index=False)
    
    #criando wordcloud
    dict_freq = {}
    for i,row in df.iterrows():
        dict_freq[row["palavra"]] = row["relevancia"]
        
    dict_freq_o = {}
    for i,row in df_o.iterrows():
        dict_freq_o[row["palavra"]] = row["relevancia"]
        
    dict_freq_n = {}
    for i,row in df_n.iterrows():
        dict_freq_n[row["palavra"]] = row["relevancia"]
    
    wordcloud_all = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(frequencies=dict_freq)
    wordcloud_all.to_file("wordcloud_all.png")
    
    wordcloud_oreo = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(frequencies=dict_freq_o)
    wordcloud_oreo.to_file("wordcloud_oreo.png")
    
    wordcloud_negresco = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(frequencies=dict_freq_n)
    wordcloud_negresco.to_file("wordcloud_negresco.png")
    
    
    
    

    