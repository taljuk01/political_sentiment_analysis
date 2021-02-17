___
<p align="center">
   <b>ANALISIS DE SENTIMIENTOS POLÍTICOS EN TWITTER</b><br><br>
   Autores:<br> 
   Lain Taljuk (aportes al código)<br>
   Francesco Angeli (aportes al código)<br>
   Juan Manuel Tabuenca (aportes a la extracción de insights)<br>
   Franco Benites (aportes a la extracción de insights)<br>
</p>

___
## Introducción
En este trabajo se aplicaron técnicas de aprendizaje automático (Machine Learning) en combinación con Natural Language Processing a un conjunto de miles de tweets recolectados previamente a las fechas de la elecciones del 11/08/2019 y del 27/10/2019, con el fin de categorizarlos de modo automático bajo las etiquetas positivo, negativo o ambiguo y, de este modo, intentar brindar una respuesta a la inquietud de si los mensajes intercambiados en esta plataforma online reflejan verosímilmente sentimientos acerca de la política manifestados de manera offline. Quedaría planteada la interrogante en vistas a futuras investigaciones, si un análisis como el de este trabajo, representa un alternativa costo-eficiente y más precisa a las encuestadoras tradicionales.
___
Para los que no tengan ganas de leer código, adjuntamos un link hacia nuestra publicación científica:
- [Paper: ¿Que pasó en las PASO? Descifrando el sentimiento político con Machine Learning](https://drive.google.com/open?id=1qHfM_APQ4BTpzUdQc7X_51RGU5Mpi5De)
___

Los interesados en el código pueden seguir deslizando hacia abajo:

### Prerequisitos
- Conocimientos de estructuras de datos y bucles condicionales con Python (listas, tuplas,diccionarios, ciclo for, if)
- Expresiones regulares (Módulo Regex)
- Conocimientos de técnicas de Natural Language Processing (framework Spacy y NLTK, wordclouds)
- Scikit Learn (Machine Learning, algoritmos supervisados, precisión, recall, f1-score)
- Pandas, Numpy, MatplotLib
- PCA (principal component analysis)
- Web Scrapping

### Instalación de paquetes necesarios utilizando pip
- pip install GetOldTweets3 (paquete que utilizaremos para scrappear los tweets)
- pip install pandas 
- pip install XlsxWriter (instalar creador de archivos excel)
- pip install numpy 
- pip install matplotlib 
- pip install scikit-learn 
- pip install nltk 
- pip install -U spacy
- pip install -U spacy-lookups-data
- python -m spacy download en_core_web_sm
- python -m spacy download es_core_news_sm
- pip install wordcloud (para crear wordclouds)
- pip install pillow
- pip install image

### Instalación de paquetes para los usuarios de miniconda
- conda install  jupyter
- conda install  ipython
- conda install  numpy
- conda install  pandas
- conda install  matplotlib
- conda install  scikit-learn
- conda install  nltk
- dentro de ipython ejecutar nltk.download() y descargar: punkt, porter_test, snowball_data, spanish_grammars 
- python -m nltk.downloader stopwords
- conda install spacy
- python -m spacy download es_core_news_sm (palabras español)
- python -m spacy download en_core_web_sm
- conda install xlsxwriter
- pip install GetOldTweets3 (este paquete no spoporta el comando con conda install)
- conda install wordcloud
- conda install pillow
- pip install image (no encontramos como instalarlo usando conda)
- conda install -c anaconda py-xgboost


### Armando el Workflow

- Sección A: Obteniendo los datos 
   - Paso A.1: Scrappear los tweets de una semana previa a las PASO 2019 (dataset sobre el cual se realizan las predicciones)
   - Paso A.2: Eliminando menciones conjuntas
   - Paso A.3: Crear un corpus propio de entrenamiento (dataset que se usará para entrenar el modelo)
- Sección B: Preprocesamiento del conjunto de entrenamiento
- Sección C: División del dataset de entrenamiento (80% para entrenar, 20% para testing)
- Sección D: Clasificador OneVsRest Clasiffier Linear SVC
   - Paso D.1: Construcción del vector de características (Tfidf Vectorizer)
   - Paso D.2: Entrenamiento del modelo
   - Paso D.3: Evaluación de performance
   - Paso D.4: Uso y prueba del modelo
- Sección E: Extracción de Insights
   - Paso E.1: Metodo del vote share para predecir elecciones 
   - Paso E.2: Predicción de los resultados electorales del 27 de octubre (obvio sin conocer los datos oficiales)
   - Paso E.3: Creación de una Wordcloud (nube de palabras, o palabras con mas frecuencia en los tweets)
 - Conclusiones
      
### Antes de empezar definamos lo siguiente: ¿Que es el analisis de sentimientos?

Se refiere al uso del procesamiento del lenguaje natural para determinar la actitud, opiniones y emociones de una persona cuando habla sobre un tema determinado (política, deporte, cine, etc).
Esencialmente, es el proceso de determinar si un escrito es positivo o negativo. 
Un uso común de esta tecnología proviene de su despliegue en el espacio de las redes sociales para descubrir cómo se sienten las personas sobre ciertos temas, particularmente a través del boca a boca de los usuarios en publicaciones textuales.


### Sección A: Obtención de los datos
#### A.1 Scrapping de tweets

Como ya se mencionó anteriormente, los datos serán extraidos desde Twitter. En esta oportunidad se utilizó el framework GetOldTweets3 por el siguiente motivo:

- La API oficial de Twitter tiene una limitacion de tiempo que no permite obtener tweets con más de una semana de antiguedad desde la fecha actual.

Este framework permite buscar los tweets más antiguos, sin limitarnos a solo una semana de búsquedas (tengamos en cuenta que este trabajo se comenzó en septiembre, y las PASO 2019 se llevaron a cabo el 11 de agosto).

Se utilizaron como filtro las siguientes palabras  para obtener los tweets de una semana previa a las paso:

**FRENTE DE TODOS**
- @cfkargentina
- @alferdez
- fernandez-fernandez
- @frentedetodos
- alberto fernandez

**JUNTOS POR EL CAMBIO**
- macri
- pichetto
- #JUNTOSPORELCAMBIO
- @MAURICIOMACRI
- @MIGUELPICHETTO
- @JUNTOSCAMBIOAR

**CONSENSO FEDERAL**
- @RLavagna 
- #consensofederal
- urtubey

**FRENTE DESPERTAR**
- Espert
- #Despertar
- @jlespert
- @luisrosalesARG
- @FrenteDespertar
- #DejenCompetirAEspert

**FIT UNIDAD**
- Nicolas del caño
- #FITunidad
- @FTE_izquierda
- @NicolasdelCano
- #YoVoteFITUnidad
- @RominaDelPla


**FRENTE NOS**
- @juanjomalvinas
- #FrenteNOS

Si bien en algunos partidos utilizamos menor cantidad de palabras filtro para obtener los tweets, en todos los casos se limitó la busqueda a solo 21 mil tweets por partido, de modo contrario el scrapping podría haber durado horas (de esta manera la cantidad máxima de tweets que pueden obtenerse es 126000).  
Entonces si queremos obtener 21 mil tweets por partido del día previo a las PASO 2019:

- Si el partido cuenta con 6 palabras filtro, entonces se necesitan 21000/6= 3500 tweets por cada palabra filtro.
- Entonces se deben scrapear 3500 toptweets (tweets con mayor cantidad de interacciones) por cada palabra filtro.

Existen muchos criterios para obtener los tweets, pero según el enfoque dado en este ["Paper de Metexas"](https://www.researchgate.net/publication/220876090_How_Not_to_Predict_Elections) se obtienen mejores resultados solo con información del día previo a las elecciones. Lo ideal sería dejar la pc toda la noche scrapeando la mayor cantidad posible de tweets, pero no contábamos con los recursos de Hardware mas óptimos.
Para captar estos tweets se diseñó la siguiente función:

~~~
def scrapper(initdate, finaldate ,toptweets=False, maximo, words=[]):

    s1=[]
    lista_final=[]
    for word in words:
        try:
            s1.append(got.manager.TweetCriteria().setQuerySearch(word).setSince(initdate).setUntil(finaldate).setTopTweets(toptweets).setMaxTweets(int(maximo)))
        except: continue
    
    for i in range (len(s1)):
        lista_final.append(got.manager.TweetManager.getTweets(s1[i]))
    return lista_final
~~~

Luego se determina la lista de palabras claves para cada candidato:

~~~
res0= ['Espert', '#Despertar', '@jlespert', '@luisrosalesARG','@FrenteDespertar', '#DejenCompetirAEspert']
res1=['macri', 'pichetto', '#JUNTOSPORELCAMBIO', '@MAURICIOMACRI', '@MIGUELPICHETTO', '@JUNTOSCAMBIOAR']
res2=['@cfkargentina', '@alferdez', 'fernandez-fernandez', '@frentedetodos', 'alberto fernandez']
res3=['@RLavagna', '#consensofederal', 'urtubey']
res4=['Nicolas del caño', '#FITunidad', '@FTE_izquierda', '@NicolasdelCano', '#YoVoteFITUnidad', '@RominaDelPla']
res5=['@juanjomalvinas', '#FrenteNOS']
~~~

Despues se aplica la función "scrapper" para buscar estos tweets:

~~~
resultado0=scrapper("2019-08-10", "2019-08-11", True, 3500, res0)
resultado1=scrapper("2019-08-10", "2019-08-11", True, 3500, res1)
resultado2=scrapper("2019-08-10", "2019-08-11", True, 3500, res2)
resultado3=scrapper("2019-08-10", "2019-08-11", True, 3500, res3)
resultado4=scrapper("2019-08-10", "2019-08-11", True, 3500, res4)
resultado5=scrapper("2019-08-10", "2019-08-11", True, 3500, res5)
~~~

Finalmente se debe extraer la información de cada tweet, esto con la siguiente función:

~~~
def extractor(tweets):
    l1,l2,l3,l4,l5,l6=[],[],[],[],[],[]

    for i in range(len(tweets)):
        for tweet in tweets[i]:
            l1.append(tweet.text)
            l2.append(tweet.date)
            l2.append(tweet.username)
            l3.append(tweet.hashtags)
            l4.append(tweet.favorites)
            l5.append(tweet.retweets)
        
    final=pd.DataFrame(list(zip(l1, l2,l3,l4,l5)), 
                       columns =['Tweets', 'Date','User', 'hashtags', 'Favs','RT'])

    
    return final
~~~

Aplicandola para cada partido político:

~~~
df0=extractor(resultado0)
df1=extractor(resultado1)
df2=extractor(resultado2)
df3=extractor(resultado3)
df4=extractor(resultado4)
df5=extractor(resultado5)
~~~

Y lo guardamos en un archivo de excel:

~~~
with pd.ExcelWriter(r'./27-de-octubre.xlsx', engine='xlsxwriter',options={'strings_to_urls': False}) as writer:
     
 
        df0.to_excel(writer, sheet_name='espert',index = None, header=True)
        df1.to_excel(writer, sheet_name='macri',index = None, header=True)
        df2.to_excel(writer, sheet_name='cfk',index = None, header=True)
        df3.to_excel(writer, sheet_name='lavagna',index = None, header=True)
        df4.to_excel(writer, sheet_name='centurion',index = None, header=True)
        df5.to_excel(writer, sheet_name='fit',index = None, header=True)

~~~

Probablemente excel tenga problemas con este archivo, ya que la columna de fechas se actualiza cada menos de 15 minutos, si esto ocurre, será necesario dropear esta columna antes de guardar el archivo:

~~~
dfa=df0.drop(['date'], axis=1)
dfb=df1.drop(['date'], axis=1)
dfc=df2.drop(['date'], axis=1)
dfe=df3.drop(['date'], axis=1)
dff=df4.drop(['date'], axis=1)
dfg=df5.drop(['date'], axis=1)
~~~

Y lo guardamos en un archivo de excel:

~~~
with pd.ExcelWriter(r'./5partidos.xlsx', engine='xlsxwriter',options={'strings_to_urls': False}) as writer:
     
        df0.to_excel(writer, sheet_name='espert',index = None, header=True)
        df1.to_excel(writer, sheet_name='macri',index = None, header=True)
        df2.to_excel(writer, sheet_name='cfk',index = None, header=True)
        df3.to_excel(writer, sheet_name='lavagna',index = None, header=True)
        df4.to_excel(writer, sheet_name='centurion',index = None, header=True)
        df5.to_excel(writer, sheet_name='fit',index = None, header=True)

~~~

Probablemente excel tenga problemas con este archivo, ya que la columna de fechas se actualiza cada menos de 15 minutos, si esto ocurre, será necesario dropear esta columna antes de guardar el archivo:

~~~
dfa=df0.drop(['date'], axis=1)
dfb=df1.drop(['date'], axis=1)
dfc=df2.drop(['date'], axis=1)
dfe=df3.drop(['date'], axis=1)
dff=df4.drop(['date'], axis=1)
dfg=df5.drop(['date'], axis=1)
~~~

En total se pudo scrapear un poco mas de 109 mil tweets.

#### A.2 Eliminando menciones conjuntas

Para que los resultados presenten el menor sesgo posible, fue necesario eliminar lo que se conoce como "menciones conjuntas", es decir, la mención de un candidato dentro de un dataset ajeno a su partido.
Por ej. mencionar de forma negativa al candidato Lavagna dentro del dataset de "Juntos por el Cambio" provocaría que se sume un voto negativo al candidato Mauricio Macri, lo cual no sería correcto **(una mejora a futuro será que el algoritmo sea capaz de determinar hacia que partido está dirigido el tweet)**.
Eliminar menciones conjuntas es relativamente sencillo. Primero determinamos todas las posibles formas en las que se pueda nombrar a alguno de los candidatos y/o partidos:

~~~
men_FF=['alberto','cfk','cristina','fernandez','fernández','massa','ff','alferdez'] #formas de nombrar a Alberto o a su partido
men_MM=['macri','mm','mauricio','pichetto','vidal'] #formas de nombrar a Mauricio Macri o a su partido
men_FIT=['caño','nico','nicolas','nicolás'] #formas de nombrar a Nicolas del Caño  o a su partido
men_espert=['espert','josé','luis','profesor','rosales'] #formas de nombrar a Espert o a su partido
men_lavagna=['lavagna','roberto','urtubey'] #formas de nombrar a Lavagna o a su partido
men_cent=['gomez','centurion','centurión'] #formas de nombrar a Gomez Centurión o a su partido
~~~

Luego creamos una lista para cada partido, en la cual se almacenan todas formas de mencionar a otro partido y/o candidato:

~~~
others_M= men_FF +  men_FIT + men_espert + men_lavagna + men_cent # mención de otros candidatos en el dataset de Macri
others_FF= men_MM +  men_FIT + men_espert + men_lavagna + men_cent # mención de otros candidatos en el dataset de Alberto Fernandez
others_LAVG= men_FF +  men_FIT + men_espert + men_MM + men_cent # mención de otros candidatos en el dataset de Lavagna
others_FIT= men_FF +  men_MM + men_espert + men_lavagna + men_cent # mención de otros candidatos en el dataset de Nicolás del Caño
others_ESP= men_FF +  men_FIT + men_MM + men_lavagna + men_cent # mención de otros candidatos en el dataset de Espert
others_CENT= men_FF +  men_FIT + men_MM + men_lavagna + men_espert # mención de otros candidatos en el dataset de Gomez Centurión
~~~

Finalmente creamos una función para eliminar menciones conjuntas, la cual deberá aplicarse a cada dataset:

~~~
def eliminador(tweet,nombres):
    temp=str(tweet)
    
    temp= re.sub(r'\W+', ' ',temp) #elimino simbolos, ya que aplicaremos esta función sobre los tweets sin procesar
                                   # mas abajo se explica el por qué de esto.
                                   
    tokens=word_tokenize(temp) #tokenizamos
    for word in tokens:
        for nombre in nombres:
            x = re.findall(nombre, word.lower()) 
            if x :
                empty=''
                return empty # si encuentra la mención un candidato ajeno al partido, se elimina el tweet
            else:
                continue # continuamos hasta encontrar una coincidencia
    return tweet #si no se encuentra una mención de otro candidato, se retorna el tweet original
~~~

Y luego aplicamos la función sobre los 6 datasets predecidos en las secciones previas. Por ejemplo, si queremos eliminar menciones de partidos ajenos a Juntos por el Cambio:

~~~
dfa['Tweets']=dfa["Tweets"].map(lambda x : eliminador(x,others_M))  #OJO, no aplicarlo sobre columna corregido
dfa = dfa[np.isnotnull(dfa['Tweets'])] #dropeamos filas con valores nulos de la columna 'tweets' (sin procesar)
~~~

Luego de llevar a cabo esto, la cantidad final de tweets por cada partido fue:

- Frente de Todos : 17338 tweets
- Juntos por el Cambio: 29736 tweets
- Consenso Federal : 1016 tweets
- Frente Despertar : 5174 tweets
- Fit Unidad : 2254 tweets
- Frente NOS: 1032 tweets

#### A.3 Creación del corpus de entrenamiento

En la mayoría de los casos, los datos que se utilizan para entrenar un algoritmo de machine learning pueden significar el éxito o el fracaso del modelo final (y del proyecto en sí). Como el objetivo de este trabajo consiste clasificar de manera automática los tweets scrapeados según su polaridad (**positiva, negativa o ambigua**), es necesario contar con un corpus lingüístico adaptado a política
argentina para poder entrenar el modelo de machine learning.
En resumidas palabras, un corpus de entrenamiento es un conjunto de textos relativamente grande, creado independientemente de sus posibles formas o usos, con el objetivo de que sean utilizados para hacer análisis estadísticos y contrastar hipótesis sobre el área que
estudian. Ante la necesidad de adaptar el algoritmo a las expresiones y modismos que utilizan los argentinos al momento de hablar sobre política, se decidió armar un corpus propio de entrenamiento y no hacer uso de corpus lingüísticos internacionales, como los que
provee la sociedad española de Natural Language Processing (corpus TASS).  
(De todos modos, para los interesados se deja el link para descargarlo : [corpus TASS](http://www.sepln.org/workshops/tass/tass_data/download.php?auth=SthaMsBsw4leVvKe1r9))  
Para armar este corpus político se clasificó manualmente cerca de 3 mil tweets del total de tweets scrapeados, con las leyendas positivo, negativo y ambiguo, siempre buscando que las 3 clases se encuentren balanceadas (mil tweets clasificados por
cada sentimiento).  
Este corpus de entrenamiento permitió que la precisión final del algoritmo aumente considerablemente (pasando del 56% al 94%), como era de esperarse, ya que es un corpus mucho más adaptado al problema planteado (el mismo se halla en la carpeta datasets, es el archivo llamado "train_dataset_FINAL.xlsx")

### Sección B: Preprocesamiento del conjunto de entrenamiento

Antes de pasar clasificación de cada tweet según su polaridad, hay que realizar algunas tareas de limpieza. De hecho, este paso es crítico y generalmente toma mucho tiempo cuando se construyen modelos de Machine Learning. 
El objetivo de la limpieza de datos será reducir el contenido de cada tweet a su mínima expresión, eliminando todo lo que no contribuya a su polaridad (ya sea positiva o negativa). Por lo tanto, en cada tweet se aplicarán las siguientes acciones:

- Se convierte todo el texto a mínuscula
- Se eliminan todo tipo de caracteres no alfabéticos
- Se eliminan todo tipo de URL's 
- Se eliminan nombres de usuario, emojis, etc.
- Se detecta todo tipo de insulto o palabra despectiva y se lo transforma a la etiqueta 'toxicword', con el objetivo de unificar
en una sola palabra muchísimas expresiones de carácter despectivo, sin importar si el insulto está escrito en singular o plural, con caracteres repetidos, o dirigido hacia un hombre o una mujer (este paso no se publica por fines comerciales, cualquier cosa comunicarse al mail taljuk01@gmail.com).
- Se detecta la mención de algun candidato ya sea por nombre, apellido o apodo y se lo transforma a la etiqueta "somecandidate".
- Se eliminan 'stopwords' (palabras que se repiten con mucha frecuencia pero que no aportan demasiado valor sintáctico, como por ej: de, por, con, ....)
- Se unifican las risas : 'ajjajajajajaj' ----> 'jaja', 'jojojo'----> jaja
- Se lleva a cabo un spellchecking, es decir, se corrigen las palabras mal escritas debido a modismos en redes sociales.  
Ej: 'xq'--> 'porque' , 'q' ---> 'que', etc.
- Se eliminan frases que hagan referencia a cosas dichas por otras personas.  
Por ej: 'El candidato dijo que "tendremos un futuro mejor" ---> 'El candidato dijo que'

Entonces, con esto sabemos lo que necesitamos mantener en los tweets y lo que necesitamos sacar. Esto se aplica a los conjuntos de entrenamiento y prueba.  

Primero importemos todos los paquetes que se utilizarán a lo largo del proyecto:
~~~
#Others

import pandas as pd
import numpy as np
import re
from string import punctuation
from unicodedata import normalize


# Text processing

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.data import load
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist
import spacy

# Machine Learning

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate,GridSearchCV,train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

~~~

Ahora se deben definir  las funciones para: eliminar acentos, hacer spellchecking, unificar candidatos y limpiar el texto. Lamentablemente no será posible compartir el código de estas funciones, ya que este proyecto esta siendo actualmente utilizado en investigaciones privadas. De todos modos con las pautas antes dadas, no dudamos en que un buen programador pueda diseñar una función igual o mejor que la nuestra.  
La funcion final para procesar todo el texto se llama "text_cleaner" y recibe 3 parametros: 
- El texto del tweet  
- Un valor booleano que, si lo seteamos a True, provoca que se lleve a cabo una acción conocida como stemming. **Stemming** es el proceso por el cual transformamos cada palabra en su raiz. Por ejemplo las palabras encantado, encantada o encantados comparten la misma raíz y se consideran la misma palabra tras el stemming. Nosotros lo usamos en False ya que resultó ser contraproducente para la precisión final del modelo de machine learning (aunque para otras aplicaciones puede resultar útil).  
- Un valor booleano que permite detectar palabras que expresen emociones (amar, odiar, orgulloso, etc) o **detecar cuando se invierte una emocion** , por ejemplo "yo amo el helado" es una frase positiva, pero "yo no amo el helado" es una frase con una emoción negativa, **debido a la presencia de la palabra 'no'**, la cual invierte la polaridad de la emoción.

Finalmente aplicamos la función text_cleaner al corpus de entrenamiento diseñado (los datasets que se usaron en este proyecto pueden descargarse desde la carpeta "datasets", al inicio del repositorio):

~~~
corpus=pd.read_excel("./train_dataset_FINAL.xlsx")

corpus["corregido"]=corpus["Tweets"].map(lambda x : text_cleaner(x, False, True, True))
~~~
De este modo cada tweet queda listo para ser vectorizado. Veamos un ejemplo de como trabaja text_cleaner:
~~~
tweet= 'lavagna YO te #apoyo en estas elecciones!!!!. Todos los bobos que no te voten pueden ver este link: www.link.com'
text_cleaner(tweet, False)

Out: 'somecandidate apoyo elecciones toxicword no voten pueden ver enlace'
~~~

#### Sección C: División del dataset de entrenamiento (80% para entrenar, 20% para testing)

Si bien el corpus de entrenamiento original cuenta con 3000 tweets, con el tiempo se fueron agregando un poco más gracias al uso de nuestro clasificador, por lo que hoy en dia está compuesto por 14027 tweets de diferenets polaridades (luego estos nuevos tweets fueron testeados manualmente para comprobar su correcta clasificación).  
Se utilizará el 80% de ellos para entrenar el modelo de ML y el 20% para testing:

~~~
X_train, X_test, y_train, y_test = train_test_split(corpus.corregido, corpus.polaridad, test_size=0.20,random_state=1,shuffle=True)
~~~

### Sección D: Clasificador OneVsRest Clasiffier Linear SVC
#### D.1: Construcción del vector de características (Tfidf Vectorizer)

Para poder analizar los tweets con nuestro modelo de ML, tenemos que extraer y estructurar la información contenida en el texto de forma numérica. Existen muchas maneras de realizar esto, pero en este trabajo utilizaremos el tipo de vectorización Tfidf, que básicamente convierte el tweet procesado a un vector numérico, la cual servirá como entrada para el modelo de ML.  
Para entenderlo mejor, pueden visitar nuestro paper y ver la subsección 2.C.1 o ver el enlace "vectorización Tfidf" adjuntado abajo:
- [Paper: ¿Que paso en las PASO?](https://drive.google.com/open?id=1qHfM_APQ4BTpzUdQc7X_51RGU5Mpi5De)
- [vectorización Tfidf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).  
En la subsección D.2 de este trabajo se adjunta el código para llevar a cabo esta vectorización.

#### D.2: Entrenamiento del modelo

En esta sección es donde se selecciona el algoritmo mas adecuado y medimos su eficacia. 
La métrica elegida para evaluar el modelo fue mediante su "accuracy" (porcentaje total de elementos clasificados correctamente), ya que este es un problema de tipo "Multi-Class Clasification" y no conviene usar métricas para problemas de clasificación binaria (como curva ROC). 
Tras evaluar varios modelos, el algoritmo OneVsRest Linear SVC fue el que otorgó mejores resultados. Para comprender un poco mejor como funcionan el algoritmo SVC y la técnica One Vs Rest dejamos estos dos enlaces:
- [Técnica One Vs Rest](https://prakhartechviz.blogspot.com/2019/02/multi-label-classification-python.html)
- [Linear SVC](https://www.youtube.com/watch?v=N1vOgolbjSc&t=591s)
- [Paper: ¿Que pasó en las PASO?](https://drive.google.com/open?id=1qHfM_APQ4BTpzUdQc7X_51RGU5Mpi5De) (ver subsección 2.C.2)

~~~
pipeline1 = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True, # curva logarítmica
                                          ngram_range=(1,2), # uso de bigramas
                                          norm='l2', # se utlizan vectores con norma unitaria
                                          
                                         )),
                
                 ('clf', OneVsRestClassifier(LinearSVC(C=1,multi_class='ovr',random_state=1,tol=0.01)))
                 
            ])



pipeline1.fit(X_train, y_train) # entrenamos el modelo

prediction = pipeline1.predict(X_test) # realizamos las predicciones sobre el dataset de testing
~~~

Para entender como funciona pipeline, dejamos un enlace que lo explica muy bien: [Comprender que es Pipeline](https://medium.com/datos-y-ciencia/pipeline-python-20c84e255444).  

#### D.3: Evaluación de performance

Ahora sí, midamos la performance de nuestro modelo:
~~~
report = classification_report(y_test, prediction, output_dict=True)

print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))
print('positive: ', report['POSITIVO'])
print('negative: ', report['NEGATIVO'])
print('ambiguo: ', report['AMBIGUO'])
~~~
La salida es:  

Test accuracy is 0.9522451888809693  
positive:  {'precision': 0.8947368421052632, 'recall': 0.7338129496402878, 'f1-score': 0.8063241106719368, 'support': 139}  
negative:  {'precision': 0.9579193454120397, 'recall': 0.9767580452920143, 'f1-score': 0.9672469755089996, 'support': 1678}  
ambiguo:  {'precision': 0.9490316004077471, 'recall': 0.9413549039433772, 'f1-score': 0.9451776649746193, 'support': 989}

Bastante bien considerando la dificultad del problema.

#### D.4 Uso y prueba del modelo

Primero importamos el dataset que contiene los tweets de cada partido:

~~~
dfa=pd.read_excel("./5partidos.xlsx", sheet_name='macri')
dfb=pd.read_excel("./5partidos.xlsx", sheet_name='alberto')
dfc=pd.read_excel("./5partidos.xlsx", sheet_name='lavagna')
dfd=pd.read_excel("./5partidos.xlsx", sheet_name='fit')
dfe=pd.read_excel("./5partidos.xlsx", sheet_name='espert')
dff=pd.read_excel("./5partidos.xlsx", sheet_name='centurion')
~~~

Luego aplicamos a cada dataset la función text_cleaner:
~~~
dfa["corregido"]=dfa["Tweets"].map(lambda x : text_cleaner(x, False, True)) #macri
dfb["corregido"]=dfb["Tweets"].map(lambda x : text_cleaner(x, False, True)) #alberto
dfc["corregido"]=dfc["Tweets"].map(lambda x : text_cleaner(x, False, True)) #lavagna
dfd["corregido"]=dfd["Tweets"].map(lambda x : text_cleaner(x, False, True)) #fit
dfe["corregido"]=dfe["Tweets"].map(lambda x : text_cleaner(x, False, True)) #espert
dff["corregido"]=dff["Tweets"].map(lambda x : text_cleaner(x, False, True)) #centurion
~~~

Ahora volvemos a entrenar el modelo, pero esta vez con el 100% del dataset de entrenamiento:

~~~
pipeline1 = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True,
                                          stop_words=stopwords.words('spanish'),
                                          ngram_range=(1,2),
                                          norm='l2'
                                         )),
                
                 ('clf', OneVsRestClassifier(LinearSVC(C=1,multi_class='ovr',random_state=1,tol=0.02)))
                 
            ])



pipeline1.fit(corpus.corregido, corpus.polaridad)
~~~
Luego realizamos las predicciones en cada dataset:
~~~
dfa['predicted_polarity'] = pipeline1.predict(dfa['corregido']) #creamos una nueva columna con la predicción de la polaridad
dfb['predicted_polarity'] = pipeline1.predict(dfb['corregido'])
dfc['predicted_polarity'] = pipeline1.predict(dfc['corregido'])
dfd['predicted_polarity'] = pipeline1.predict(dfd['corregido'])
dfe['predicted_polarity'] = pipeline1.predict(dfe['corregido'])
dff['predicted_polarity'] = pipeline1.predict(dff['corregido'])
~~~

Finalmente guardamos estas predicciones un archivo de excel:

~~~
with pd.ExcelWriter(r'./PASO-2019-PREDECIDOS.xlsx', engine='xlsxwriter',options={'strings_to_urls': False}) as writer:
     
        dfa.to_excel(writer, sheet_name='macri',index = None, header=True)
        dfb.to_excel(writer, sheet_name='alberto',index = None, header=True)
        dfc.to_excel(writer, sheet_name='lavagna',index = None, header=True)
        dfd.to_excel(writer, sheet_name='fit',index = None, header=True)
        dfe.to_excel(writer, sheet_name='espert',index = None, header=True)
        dff.to_excel(writer, sheet_name='centurion',index = None, header=True)
~~~

### Sección E: Extracción de Insights
#### E.1: Metodo del vote share para predecir elecciones

La idea principal en este punto es determinar si realmente existe algún tipo de relación entre el comportamiento político de las personas en redes sociales y su comportamiento político en la vida real. Para esto, es necesario encontrar un modo de relacionar el porcentaje de tweets clasificados como positivos , negativos o ambiguos por el modelo de ML, con el resultado real que ocurrió en las elecciones provisorias en Argentina, las PASO 2019.
Como primera medida, facilitamos los siguientes trabajos de investigación, los cuales servirán para entender un poco todo lo que sigue:  
- ["A meta-analysis of electoral prediction from Twitter data"](https://arxiv.org/ftp/arxiv/papers/1206/1206.5851.pdf)
- ["How (Not) To Predict Elections"](https://www.researchgate.net/publication/220876090_How_Not_to_Predict_Elections)

Siguiendo los estudios expuestos por en los dos links anteriores, se pueden hallar dos métodos interesantes de predicción:
1) El primer método de predicción consiste solamente en contar el número de tweets que mencionan a cada candidato. Según estos
estudios, la proporción de tweets que mencionan a cada candidato debería reflejar de cerca la participación real de los votos en las elecciones.
2) El segundo método de predicción aprovecha la naturaleza bipartidista de un proceso electoral (como es el caso de Argentina), y propone un método conocido como "vote-share", donde se considera que la mayor parte de los votos de la elección estarán dirigidos solamente hacia los dos candidatos más populares:

![](/imagenes/formula-vote-share.png)

Algunas cuestiones sobre las fórmulas anteriores:
- Se etiquetan los 2 candidatos más populares como C1 y C2 respectivamente.
- vote_share(C1) es la fórmula que computa los votos para el candidato C1.
- vote_share(C2) es la fórmula que computa los votos para el candidato C2.
- Pos(C1) y Neg(C1) son, respectivamente, el número de tweets positivos y negativos que mencionan al candidato C1.
- Pos(C2) y Neg(C2) son, respectivamente, el número de tweets positivos y negativos que mencionan al candidato C2.

Particularmente para este trabajo, el segundo método entregó mejores resultados, ya que refleja mejor el clima electoral argentino.
Hay que tener en cuenta que comúnmente en una carrera electoral participan más de dos candidatos, por lo que es necesario normalizar los resultados de las PASO 2019 para que puedan alcanzar el 100%. Finalmente se compara este resultado normalizado con aquél del vote-share para sacar conclusiones.  
En la siguiente tabla se muestra el porcentaje real de votos de las PASO 2019 hacia cada candidato:

![](/imagenes/resultados-reales-paso.png)

Claramente se observa que Mauricio Macri y Alberto Fernández fueron los dos candidatos más populares de las elecciones primarias. Aunque Roberto Lavagna también obtuvo una buena cantidad votos, para este trabajo se consideró despreciable su porcentaje con respecto a los dos primeros candidatos, ya que la idea es utilizar el método del vote-share, el cual resulta válido para elecciones bi-partidistas o muy polarizadas. Normalizando los votos que obtuvieron los dos candidatos más populares se obtiene:

![](/imagenes/normalizados-PASO-2019.png)

Ahora es necesario estudiar los resultados que arrojó el algoritmo de ML con la técnica de sentiment analysis. Los mismos se adjuntan en la tabla de abajo, indicando la cantidad de votos positivos y negativos hacia cada candidato (los tweets ambiguos fueron descartados ya que no representan una intención de voto definida):

![](/imagenes/sentiment-PASO-2019.png)

Ahora se aplica el vote share para ambos candidatos (la sigla AF representa al candidato Alberto Fernández y la sigla MM al candidato Mauricio Macri ):

![](/imagenes/vote-share-PASO-2019.png)

Es posible observar una gran similitud entre el porcentaje de votos normalizados de las PASO y el porcentaje de votos calculados por el vote-share. Por lo tanto, el análisis de sentimiento logró explicar con un margen de error de aproximadamente % 2,5 el resultado normalizado de las PASO.
Si bien el candidato Mauricio Macri obtuvo mayor cantidad de tweets positivos que Alberto Fernández, la cantidad de tweets negativos que recibió fue muchísimo mayor en comparación con la del candidato del Frente de Todos.
Resulta interesante como una red social como Twitter pudo reflejar tan bien los resultados de las PASO 2019,
por lo que en la siguiente sección se aplicará el mismo método con el objetivo de dar una predicción para las elecciones del 27 de octubre.  

*Nota: esta predicción fue llevada a cabo antes de la escritura de este trabajo, con 2 hs de anticipación a los resultados oficiales vistos en televisión. De todos modos, aquí se hará uso de los resultados oficiales para compararlos con las predicciones dadas.*

#### E.2: Predicción de los resultados electorales del 27 de octubre

Para esta última sección del trabajo se aplicó todo lo visto anteriormente, con el objetivo de dar una predicción de las elecciones del 27/10/2019 haciendo uso solamente de Twitter en combinación con sentiment analysis y machine learning.
Primero se hizo scraping de Tweets con 48 hs previas a las elecciones (con el fin de intentar captar más tweets). Para captar estos tweets se aplicó la misma función que al comienzo de este trabajo, solo que se quitó la parte de setMaxTweets con el fin de captar la mayor cantidad posible de tweets:

~~~
def scrapper(initdate, finaldate ,toptweets=False, words=[]):

    s1=[]
    lista_final=[]
    
    for word in words:
        try:
            s1.append(got.manager.TweetCriteria().setQuerySearch(word).setSince(initdate).setUntil(finaldate).setTopTweets(toptweets))
        except: continue
    
    for i in range (len(s1)):
        lista_final.append(got.manager.TweetManager.getTweets(s1[i]))
    return lista_final
~~~

Luego se determinaron la lista de palabras claves para cada candidato:

~~~
res0=['@jlespert', 'Espert', '@FrenteDespertar']
res1=['@mauriciomacri', 'macri', 'pichetto', '@JUNTOSCAMBIOAR']
res2=['@alferdez', '@CFKArgentina', '@frentedetodos', 'cfk']
res3=['@RLavagna', '#consensofederal','urtubey']
res4=['@juanjomalvinas','#FrenteNOS']
res5=['@NicolasdelCano','#FITunidad']
~~~


Finalmente se aplicó la función "scrapper" para buscar estos tweets:

~~~
resultado0=scrapper("2019-10-25", "2019-10-27", True, res0)
resultado1=scrapper("2019-10-25", "2019-10-27", True, res1)
resultado2=scrapper("2019-10-25", "2019-10-27", True, res2)
resultado3=scrapper("2019-10-25", "2019-10-27", True, res3)
resultado4=scrapper("2019-10-25", "2019-10-27", True, res4)
resultado5=scrapper("2019-10-25", "2019-10-27", True, res5)
~~~

Finalmente se debe extraer la información de cada tweet, esto se hacia con la siguiente función:

~~~
def extractor(tweets):
    l1,l2,l3,l4,l5,l6=[],[],[],[],[],[]

    for i in range(len(tweets)):
        for tweet in tweets[i]:
            l1.append(tweet.text)
            l2.append(tweet.date)
            l2.append(tweet.username)
            l3.append(tweet.hashtags)
            l4.append(tweet.favorites)
            l5.append(tweet.retweets)
        
    final=pd.DataFrame(list(zip(l1, l2,l3,l4,l5)), 
                       columns =['Tweets', 'Date','User', 'hashtags', 'Favs','RT'])

    
    return final
~~~

Entonces, extraemos la información para cada partido político:

~~~
df0=extractor(resultado0)
df1=extractor(resultado1)
df2=extractor(resultado2)
df3=extractor(resultado3)
df4=extractor(resultado4)
df5=extractor(resultado5)
~~~

Y lo guardamos en un archivo de excel:

~~~
with pd.ExcelWriter(r'./27-de-octubre.xlsx', engine='xlsxwriter',options={'strings_to_urls': False}) as writer:
     
 
        df0.to_excel(writer, sheet_name='espert',index = None, header=True)
        df1.to_excel(writer, sheet_name='macri',index = None, header=True)
        df2.to_excel(writer, sheet_name='cfk',index = None, header=True)
        df3.to_excel(writer, sheet_name='lavagna',index = None, header=True)
        df4.to_excel(writer, sheet_name='centurion',index = None, header=True)
        df5.to_excel(writer, sheet_name='fit',index = None, header=True)

~~~

Probablemente excel tenga problemas con este archivo, ya que la columna de fechas se actualiza cada menos de 15 minutos, si esto ocurre, será necesario dropear esta columna antes de guardar el archivo:

~~~
dfa=df0.drop(['date'], axis=1)
dfb=df1.drop(['date'], axis=1)
dfc=df2.drop(['date'], axis=1)
dfe=df3.drop(['date'], axis=1)
dff=df4.drop(['date'], axis=1)
dfg=df5.drop(['date'], axis=1)
~~~

Finalmente se eliminaron menciones conjuntas, se aplicó el preprocesamiento a todos los tweets y se utilizó el modelo de ML para etiquetar cada tweet según su polaridad, obteniendo los siguientes resultados (estos pasos ya son repetitivos, asi que podrás hacerlos solo):

![](/imagenes/vote-share-27-oct.png)

Con estos resultados se observa que, en comparación con las elecciones primarias del 11 de agosto, el candidato Mauricio Macri reduce la diferencia de votos con respecto a Alberto Fernández, pero no cuenta con la cantidad de votos necesaria para llegar a un ballotage.
Este resultado fue calculado el domingo 27/10/2019 a horas 19:23, antes de conocerse los resultados oficiales de las elecciones.  

Luego de publicarse los escrutinios finales, se pudo llevar a cabo la comparación. En la siguiente tabla se adjuntan los resultados reales ocurridos en las elecciones del 27/10/2019:

![](/imagenes/27-oct-prediccion.png)

Esto significa que con la técnica de sentiment analysis se pudo dar una predicción de los resultados finales con un error absoluto de aproximadamente 7%, al mismo tiempo que fue posible predecir que el candidato Mauricio Macri reduciría la diferencia de votos con respecto al candidato del Frente de Todos, Alberto Fernández.


#### E.3: Creación de una Wordcloud
Una nube de palabras o WordCloud es una representación visual de las palabras que conforman un texto, en donde el tamaño es mayor para las palabras que aparecen con mayor frecuencia.  
Para generar la wordcloud utilizaremos una hoja del dataset "5partidos.xlsx" llamada "Todos unidos", en la cual se encuentran los tweets de todos los candidatos juntos.
~~~
todos=pd.read_excel("./5partidos.xlsx", sheet_name='todos unidos')
~~~

Ahora aplicamos la función text_cleaner (esto tarda unos minutos, hay que tener en cuenta que están todos los datasets unidos):
~~~
todos["corregido"]=todos["Tweets"].map(lambda x : text_cleaner(x, False))
~~~

La idea ahora es crear un archivo .txt para almacenar en él todos los tweets procesados anteriormente (sería como armar un string enorme). 
De este modo, si necesitamos editar la wordcloud en un futuro,  nos ahorramos paso de procesar nuevamente todos los datos:
~~~
text = " ".join(tweet for tweet in todos["corregido"])
~~~

Y posteriormente lo guardamos en nuestro directorio de trabajo:

~~~
with open("todos_procesados.txt", "w",encoding="utf-8") as text_file:
     text_file.write(text)
~~~

Listo! ya tenemos en  un archivo .txt todos nuestros tweets procesados. Ahora lo podemos abrir para trabajar con él:  

~~~
with open('todos_procesados.txt', 'r',encoding="utf-8") as myfile:
     data = myfile.read()
~~~

Supongamos que queremos saber las palabras que mas se repitieron en todo el dataset procesado. Para lograr esto, hacemos uso de FreqDist:
~~~
from nltk.probability import FreqDist

def word_counter(texto,k):    #k es un numero entero, que sirve para devolver las "k" palabras mas repetidas
    int(k)
    fdist = FreqDist()

    temp=str(texto).split()
    for word in temp:
            fdist[word] += 1
    return list(fdist.most_common(k))
~~~

Supongamos que queremos conocer las 50 palabras con mayor frecuencia. Invocamos la función con k=50

~~~
word_counter(data, 50)
~~~
Y obtenemos:
~~~
[('somecandidate', 60280), ('no', 35659), ('toxicword', 15409),   ('si', 10211), 
('fernández', 6050), ('votar', 5768), ('voto', 5207), ('vos', 4461), ('menos', 3842), 
('presidente', 3818), ('va', 3432), ('ser', 3409), ('gente', 3345), ('vamos', 3254), 
('juntosporelcambio', 3226), ('argentina', 3182), ('años', 3089), ('domingo', 3042), 
('mas', 3007), ('país', 2985), ('ahora', 2635), ('bien', 2512), ('campaña', 2491), 
('frente', 2374), ('gobierno', 2371), ('kirchnerismo', 2314), ('hace', 2314), 
('mejor', 2271), ('centurión', 2254), ('puede', 2234), ('mañana', 2222), ('solo', 2126), 
('gracias', 2097), ('van', 2055), ('así', 2018), ('hacer', 2004), ('siempre', 1887), 
('paso', 1864), ('vida', 1804), ('nunca', 1775), ('hoy', 1766), ('gómez', 1745), 
('vota', 1677), ('cambio', 1628), ('ver', 1627), ('vez', 1563), ('voy', 1514), ('mismo', 1509), 
('tan', 1464), ('futuro', 1461)]
~~~

Obviamente  'toxicword' y 'somecandidate' son las palabras con mas frecuencia, ya que en la primera se engloban todo tipo de insultos y en la segunda toda mención de algun candidato. De todos modos, al momento de crear nuestra nube de palabras no las tendremos en cuenta (ya veremos como se las excluye).  
La wordcloud se puede crear de cualquier forma (por defecto se crea con forma cuadrada). Aquí haremos una wordcloud con forma de nube,pero en caso de querer una wordcloud de forma mas tradicional, podemos saltear este paso y crear la wordcloud sin usar una máscara. Para generar la forma de nube, necesitamos una imagen que sirva como máscara, en nuestro caso usaremos esta:


![cloud](https://i.ya-webdesign.com/images/clouds-animated-png-6.png)


Descargamos la imagen y la guardamos en nuestro directorio de trabajo con el nombre "cloud.png". Ahora toca vectorizar la imagen, esto es muy sencillo haciendo uso de numpy:

~~~
from PIL import Image, ImageDraw
mask1 = np.array(Image.open("cloud.png")) #vectorizamos la imagen
~~~
En caso de que queramos directamente importar la imagen desde internet ,podemos hacer lo siguiente:
~~~
import urllib
import requests
mask1 = np.array(Image.open(requests.get('https://i.ya-webdesign.com/images/clouds-animated-png-6.png', stream=True).raw))
~~~

De todos modos en el directorio de este trabajo agregamos una carpeta llamada "imagenes",donde incluimos la máscara que estamos usando.

Ahora podemos empezar a crear nuestra wordcloud. Antes que nada, si nunca trabajaste con wordclouds, aqui hay un excelente artículo:  [Generating Wordclouds](https://www.datacamp.com/community/tutorials/wordcloud-python).  
Primero definimos algunos aspectos importantes referidos a la estética de nuestra wordcloud:

- Background color
- min_font_size = 5 --> tamaño de letra mínimo 
- max_font_size=300 ---> tamaño de letra máximo (no abusar de este valor, sino las palabras mas frecuentes serán enormes)
- max_words=300 ---> cantidad máxima de palabras que aparecerán en la wordcloud
- mask=mask1 ---> si queremos que la wordcloud tenga una forma específica usamos este valor, sino no lo agregamos  
                  (mask1 la creamos en el paso anterior)
- contour_width=0.05 -----> ancho del contorno de la wordcloud
- contour_color='white' -----> color del contorno de la wordcloud
- width = 1280, height = 1240 ----> resolución de la imagen

Teniendo en cuenta todo lo anterior, ya podemos diseñar nuestra wordcloud.

~~~
wordcloud = WordCloud(width = 1280, height = 1240,
                      background_color ='white',
                      min_font_size = 5,
                      max_font_size=300,
                      max_words=300,
                      mask=mask1, # si queremos una wordcloud cuadrada tradicional, no hace falta incluir este parámetro
                      contour_width=0.05, 
                      contour_color='white',
                      stopwords=stop_w)

wordcloud.generate(data) #generamos la wordcloud

plt.figure(figsize=(10,8),facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

wordcloud.to_file("./cloud.png") # salvamos la wordcloud en el directorio de trabajo
~~~
Y nos queda algo como esto:

![](/imagenes/cloud1.png)

Si queremos quitar palabras de la wordcloud, una solución sería agregar las palabras no deseadas a la lista de stopwords. Recordemos que al principio de este proyecto, las stopwrods se habian definido del siguiente modo:

~~~
stop_words= stopwords.words('spanish')
del stop_words[15] #elimino la palabra "no" de mis stopwords

simbolos = list(punctuation)
simbolos.extend(['¿', '¡','..','...','....','.....','…','``','•',"''",'“','”',"'"]) #agrego estos simbolos
simbolos.extend(map(str,range(10))) #agrego numeros del 1 al 10 a la lista de simbolos

stop_w = set(stop_words + simbolos + ['https','http','sos'])
~~~

Si por ejemplo queremos quitar las palabras 'toxicword' y 'somecandidate' de nuestra wordcloud, simplemente las sumamos a la lista de stopwords de la siguiente manera:

~~~
stop_w = set(stop_words + simbolos + ['https','http','sos','toxicword','somecandidate'])
~~~

Y volvemos a generar la wordcloud:

~~~
wordcloud = WordCloud(width = 1280, height = 1240,
                      background_color ='white',
                      min_font_size = 5,
                      max_font_size=300,
                      max_words=300,
                      mask=mask1,
                      contour_width=0.05, 
                      contour_color='white',
                      stopwords=stop_w)

wordcloud.generate(data)

plt.figure(figsize=(10,8),facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

wordcloud.to_file("./cloud_2.png") # salvamos la wordcloud en el directorio de trabajo (ponerle otro nombre para no sobrescribir)
~~~

![](/imagenes/cloud_2.png)

Y listo, tenemos una perfecta wordcloud con lo mas hablado durante la semana previa a las paso 2019.  
Sería interesante construir una Wordcloud para los tweets clasificados como positivos y otra para los tweets clasificados como negativos (por cuestiones de tiempo no llegamos a armar una wordcloud para cada polaridad).


# CONCLUSIONES

- Con este trabajo queda demostrado el potencial que pueden tener las redes sociales para realizar inferencias sobre elecciones presidenciales, pero todavía falta mucho para mejorar como para considerarlo un método válido de predicción de elecciones.
- Sería interesante llevar a cabo este mismo estudio con otras redes sociales como facebook, Instagram o diarios Online y compararlo con los resultados que arrojó la plataforma de Twitter. 
- Por otro lado, si se cuenta con las herramientas correctas de Hardware, para elecciones futuras podrían tomarse datos en streaming con la API de twitter durante varios meses previos a las elecciones, lo cual permitiría tener una muestra mucho mayor de datos y realizar un estudio más preciso.  
- Otro punto destacable de este trabajo fue el éxito obtenido en la aplicación de técnicas de NLP y Sentiment analysis gracias a la creación de un corpus político adaptado para nuestro país y el algoritmo de procesamiento texto diseñado. Incluso la metodología de clasificación mediante NLP y procesamiento de texto, es aplicable a muchas otras ramas fuera del ámbito político como ser: polaridad de
reviews (actitud de la gente con respecto a un producto o marca), categorización de problemas en una empresa a través de comentarios que dejan los clientes, motores de recomendación de productos, chatbots, etc.  

Algunas mejoras futuro para este proyecto podría ser:
-La inclusión de politólogos, sociólogos y personas estudiosas de la lengua española para mejorar el corpus de entrenamiento de modo que resulte útil para estudios futuros, como así también buscar alternativas al método del vote-share, de modo que puedan ser incluidos todos los candidatos.
- Mejorar el algoritmo de modo que sea capaz de detectar sarcasmo, cambios en el lenguaje, hacia qué candidato o partido está dirigido un mensaje.
- Hallar algún tipo significado político a los mensajes etiquetados como negativos para candidatos considerados con menor captación de voto, como así también los tweets etiquetados como ambiguos (¿hacia quien podrían ir dirigidos esos votos?¿Por qué?).  

**No queda duda que cada vez son más los usuarios que interactúan en redes sociales, por lo que en el mediano plazo el Big Data podría competir de cerca con las encuestadoras tradicionales.**



