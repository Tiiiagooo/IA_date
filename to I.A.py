#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ouverture et lecture des json
import nltk
import spacy
import json
import tqdm
import spacy
import glob
import re
import os
def ouvrir_json(chemin):
    f = open(chemin, encoding="UTF-8")
    toto = json.load(f)
    f.close()
    return toto
def ecrire_json(chemin, contenu):
    with open(chemin, "w") as f:
        f.write(json.dumps(contenu, indent=2))
        f.close()
def ecrire_fichier(chemin, contenu):
  w = open(chemin, "w", encoding="utf-8")
  w.write(contenu)
  w.close()
def lire_fichier(chemin):
  f = open(chemin, "r", encoding="utf-8")
  chaine = f.read()
  f.close()
  return chaine
def Splittxt(txt):
    tokenizer = nltk.RegexpTokenizer(r"(\w+-\w+|\w+\S|(\w\.)*\w.|\w+|\S|\w+\S|\?|\!)")
    txt_split = tokenizer.tokenize(txt)
    return 
def Splittxt4(txt):
    tokenizer = nltk.RegexpTokenizer(r"(\w+-\w+|\w+\S|((\w\.)*\w.)|\w+|\S|\w+\S|\?|\!)")
    txt_split = tokenizer.tokenize(txt)
    return 
def Splittxt3(txt):
    tokenizer = nltk.RegexpTokenizer(r"(\w[\w\.]{1,}|\w+-\w+|\w+\S|\w+|\S|\w+\S|\?|\!)")
    txt_split = tokenizer.tokenize(txt)
    return txt_split
def Splittxt2(txt):
    tokenizer = nltk.RegexpTokenizer(r"(\w+'|\w+-\w+|\w+|\S|\w+\S)")
    txt_split = tokenizer.tokenize(txt)
    return txt_split

chemin = glob.glob('lieux_paris/*/*')
vrai_entite_nomme = []
dic_lieu = {}
for i in chemin:
    lieux_de_paris = ouvrir_json(i)
    for lieu in lieux_de_paris:
        #print(lieu)
        vrai_entite_nomme.append(lieu)


# In[2]:


corpus = ouvrir_json("tmp.json")


# In[61]:


dictionnaire_ia2 = {}
for nom_artiste ,liste_chansons in tqdm.tqdm(corpus.items()):
    for element in liste_chansons:
        date = element["release_date"]
        if date is None:
            date = "None"
        titre = element["full_title"]
        texte = element["lyrics"]
        try:
            artiste = element["album"]["artist"]["name"]
        except:
            artiste = "pas de nom"
        dictionnaire_ia2.setdefault(date[:4], {"Artiste":artiste, "Titre":titre, "Lyrics":texte})
        dictionnaire_ia2[date[:4]]["Artiste"] = artiste
        dictionnaire_ia2[date[:4]]["Titre"] = titre
        dictionnaire_ia2[date[:4]]["Lyrics"] = texte
#print(dictionnaire_ia)
ecrire_json("dictionnaire_ia2.json", dictionnaire_ia2)


# In[44]:


dictionnaire_ia = {}
for nom_artiste ,liste_chansons in tqdm.tqdm(corpus.items()):
    for element in liste_chansons:
        titre = element["full_title"]
        texte = element["lyrics"]
        date = element["release_date"]
        try:
            artiste = element["album"]["artist"]["name"]
        except:
            artiste = "pas de nom"
        dictionnaire_ia.setdefault(date, {"Artiste":"", "Titre":"", "Lyrics":""})
        dictionnaire_ia[date]["Artiste"] = artiste
        dictionnaire_ia[date]["Titre"] = titre
        dictionnaire_ia[date]["Lyrics"] = texte
#print(dictionnaire_ia)
ecrire_json("dictionnaire_ia.json", dictionnaire_ia)


# In[62]:


dictionnaire_ia3 = {}
for nom_artiste ,liste_chansons in tqdm.tqdm(corpus.items()):
    for element in liste_chansons:
        titre = element["full_title"]
        texte = element["lyrics"]
        date = element["release_date"]
        try:
            artiste = element["album"]["artist"]["name"]
        except:
            artiste = "pas de nom"
        dictionnaire_ia3.setdefault(date, [])
        if texte not in dictionnaire_ia3:
            dictionnaire_ia3[date].append([artiste, titre, texte])
ecrire_json("dictionnaire_ia3.json", dictionnaire_ia3)


# In[ ]:


dictionnaire_ia4 = {}
for nom_artiste ,liste_chansons in tqdm.tqdm(corpus.items()):
    for element in liste_chansons:
        date = element["release_date"]
        if date is None:
            date = "None"
        titre = element["full_title"]
        texte = element["lyrics"]
        try:
            artiste = element["album"]["artist"]["name"]
        except:
            artiste = "pas de nom"
        dictionnaire_ia4.setdefault(date[:4], [])
        if texte not in dictionnaire_ia3:
            dictionnaire_ia4[date[:4]].append([artiste, titre, texte])
ecrire_json("dictionnaire_ia4.json", dictionnaire_ia4)


# In[ ]:


frequence_texte = {}
for date, valeur in sorted(dictionnaire_ia4.items(), key=lambda x: x[0], reverse=False):
    for i in valeur:
        artiste = i[0]
        titre = i[1]
        texte = i[2]
        


# # Creation du corpus avant de mettre vecteur.

# In[186]:


dictionnaire_ia5 = {}
for nom_artiste ,liste_chansons in tqdm.tqdm(corpus.items()):
    for element in liste_chansons:
        date = element["release_date"]
        if date is None:
            date = "None"
        titre = element["full_title"]
        texte = element["lyrics"]
        
        if len(str(texte))> 200:
            texte = element["lyrics"]
        else:
            continue
        try:
            artiste = element["album"]["artist"]["name"]
        except:
            artiste = "pas de nom"
        #if isinstance(date, int) == True:
            #if date[:3] >= 194:
        
        if date !="None" and int(date[:4]) >= 1940:
                dictionnaire_ia5.setdefault(date[:3], {"nbre_de_texte" : 0, "nbre_de_mot" : 0, "nbre_de_caractere" : 0})
                dictionnaire_ia5[date[:3]].setdefault("info_titre", [])
                dic_tempo = {"Titre": titre, "Auteur":artiste}
                dictionnaire_ia5[date[:3]]["info_titre"].append(dic_tempo)
                dictionnaire_ia5[date[:3]].setdefault("chanson", [])
                dictionnaire_ia5[date[:3]]["chanson"].append(texte)
                if texte != None:
                    dictionnaire_ia5[date[:3]]["nbre_de_texte"] += 1
                    mot_split = Splittxt3(texte)
                    nbre_carac = 0
                    for mot in mot_split:
                        for caractere in mot:
                            nbre_carac += 1
                    dictionnaire_ia5[date[:3]]["nbre_de_mot"] += len(mot_split)
                    dictionnaire_ia5[date[:3]]["nbre_de_caractere"] += nbre_carac

                #print(len(mot_split))
                #break
            #break

            #dictionnaire_ia5[date] = { titre : {}}
    #print(dictionnaire_ia5)
ecrire_json("dictionnaire_ia5.json", dictionnaire_ia5)


# In[188]:


dictionnaire_ia6 = {}
for date, valeur in tqdm.tqdm(sorted(dictionnaire_ia5.items(), key=lambda x: x[0], reverse=False)):
    dictionnaire_ia6[date] = valeur
#print(dictionnaire_ia6)
ecrire_json("dictionnaire_ia6.json", dictionnaire_ia6)


# # Initialisation des dates et des textes sur le vectoriseur.

# In[79]:


"nbre_de_texte": 126,
    "nbre_de_mot": 32303,
    "nbre_de_caractere": 119054,
"nbre_de_texte": 177,
    "nbre_de_mot": 44939,
    "nbre_de_caractere": 169501,


# In[189]:


to_vecteur = ouvrir_json("dictionnaire_ia6.json")
liste_paroles = []
liste_date = []
for date_decennie, dic in tqdm.tqdm(to_vecteur.items()):
    chanson = dic.get("chanson")
    for paroles in chanson:
        liste_date.append(date_decennie)
        liste_paroles.append(paroles)
        
       


# In[190]:


#from sklearn.feature_extraction.text import CountVectorizer
#V = CountVectorizer() 


# In[204]:


X = V.fit_transform(liste_paroles)
y = liste_date

from sklearn.feature_extraction.text import CountVectorizer
V = CountVectorizer(ngram_range=(1,1))
X = V.fit_transform(liste_paroles)

## séparer train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[206]:


#classifier
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

list_classifieur = [
    ["Perceptron", Perceptron(eta0=0.1, random_state=0)],
    ["Support vecteur machine sans stat", svm.SVC() ],
    ["SVM linéaire", svm.LinearSVC()], # a modifier !!!!!!!!!!
    ["Arbre de decision",DecisionTreeClassifier(max_depth=5)],
    ["Random Forest",RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)],
    #["Naive Bayes", GaussianNB()]
]
list_vectorizer = [#ajouter nlemma
    ["vectoriseur simple ", CountVectorizer() ],
    ["vectoriseur sans maj", CountVectorizer(ngram_range=(1, 1), lowercase=True)],
    ["vectoriseur ngramme", CountVectorizer(ngram_range=(1, 2))],
    ["vectoriseur ngramme2", CountVectorizer(ngram_range=(2, 2))],
    ["vectoriseur ngramme3", CountVectorizer(ngram_range=(2, 3))],
    ["stopwords",CountVectorizer(ngram_range=(1, 1), stop_words='french')],
    ["max_features",CountVectorizer(ngram_range=(1, 1), max_features = 10)],
    ["max_features2",CountVectorizer(ngram_range=(1, 1), max_features = 100)],
    ["vectoriseur sans maj et ngramme", CountVectorizer(ngram_range=(1, 2), lowercase=True)],
    ["vectoriseur sans maj et ngramme2", CountVectorizer(ngram_range=(2, 2), lowercase=True)],
    ["vectoriseur sans maj et ngramme3", CountVectorizer(ngram_range=(2, 2), lowercase=True)],
    ["vectoriseur sans maj et max_features", CountVectorizer(ngram_range=(1, 1), lowercase=True ,max_features = 10)],
    ["vectoriseur sans maj et stopwords", CountVectorizer(ngram_range=(1, 1), stop_words='french', lowercase=True)],
    ["vectoriseur ngramme et stopwords", CountVectorizer(ngram_range=(1, 2), stop_words='french')],
    ["vectoriseur ngramme et max_features", CountVectorizer(ngram_range=(1, 2), max_features = 10)],
    ["vectoriseur ngramme et max_features2", CountVectorizer(ngram_range=(1, 2), max_features = 100)],
    ["vectoriseur ngramme2 et stopwords", CountVectorizer(ngram_range=(2, 2), stop_words='french')],
    ["vectoriseur ngramme2 et max_features", CountVectorizer(ngram_range=(2, 2), max_features = 10)],
    ["vectoriseur ngramme2 et max_features2", CountVectorizer(ngram_range=(2, 2), max_features = 100)],
    ["vectoriseur sans maj, ngramme et stopwords", CountVectorizer(ngram_range=(1, 2), lowercase=True,  stop_words='french')],
    ["vectoriseur sans maj, ngramme2 et stopwords", CountVectorizer(ngram_range=(2, 2), lowercase=True,  stop_words='french')],
    ["vectoriseur sans maj, max_features et stopwords", CountVectorizer(max_features = 10, lowercase=True,  stop_words='french')],
    ["vectoriseur ngramme, max_features et stopwords", CountVectorizer(ngram_range=(1, 2), max_features = 10, stop_words='french')],
    ["vectoriseur ngramme2, max_features et stopwords", CountVectorizer(ngram_range=(2, 2), max_features = 10, stop_words='french')],
    ["vectoriseur total 1", CountVectorizer(ngram_range=(1, 2), max_features = 10, stop_words='french', lowercase=True)],
    ["vectoriseur total 1", CountVectorizer(ngram_range=(2, 2), max_features = 10, stop_words='french', lowercase=True)],
]
#-------------------------------------------------------------------------------------------------------------------------------------------
with open("resultat/DonneesClassifieur2.json", "a") as f:
    f.write("[")
    f.write("\n")
    f.close()
#-------------------------------------------------------------------------------------------------------------------------------------------
#ppn = Perceptron(eta0=0.1, random_state=0)
#ppn.fit(X_train, y_train)
#y_pred = ppn.predict(X_test)
# On fait la somme de tous les cas où la valeur dans y_test est bien trouvée dans y_pred
for nom_vectoriseur,V in tqdm.tqdm(list_vectorizer):
    X = V.fit_transform(liste_paroles)# On a bien le "X", il nous manque le "y" :
    y = liste_date
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    for nom, classifieur in tqdm.tqdm(list_classifieur):
        print("---------------------------------------------------------------- \n",nom,"et ",nom_vectoriseur, "\n")
        classifieur.fit(X_train, y_train)
        y_pred = classifieur.predict(X_test)
        # On fait la somme de tous les cas où la valeur dans y_test est bien trouvée dans y_pred
        good = (y_test == y_pred).sum()
        bad = (y_test != y_pred).sum()
        Resultat = []
        Resultat.append(str(good))
        Resultat.append(str(bad))
        print('Bons résultats %d' % good)
        print('Erreurs: %d' % bad)
        print(good/(bad+good))
#-----------------------------------------------------------------------------------------------------------
        nom_classes = ["1940","1950","1960","1970","1980","1990","2000","2010","2020"]
        report2 = classification_report(y_test, y_pred, target_names=nom_classes, digits=4)
        print(report2)
#-----------------------------------------------------------------------------------------------------------
        report = classification_report(y_test, y_pred, target_names=nom_classes, digits=4, output_dict = True)
        print(report)
        with open("resultat/classificatio_report2.json", "w") as w:
            w.write(json.dumps(report, indent=2))
#-----------------------------------------------------------------------------------------------------------
        with open("resultat/DonneesClassifieur2.json", "a") as f:
            for objet in range(0,len(Resultat),2):
                f.write(""""%s """%nom)
                f.write("et")
                f.write(""" %s" """%nom_vectoriseur)
                f.write("{")
                f.write("\n")
                f.write(""""Bons résultats" :""")
                f.write(Resultat[objet])
                f.write(",")
                f.write("\n")
                f.write(""""Erreurs" :""")
                f.write(Resultat[objet+1])
                f.write(",")
                f.write("\n")
                f.write("}")
                f.write(",")
                f.write("\n")
                with open("resultat/2report_classifier="+nom+"_dataset=date.txt", "a") as w:
                    w.write(""""%s """%nom)
                    w.write(""" %s" """%nom_vectoriseur)
                    w.write("\n")
                    w.write(report2)
                    w.write("-"*200)
                    w.write("\n")
        matrice_confusion = confusion_matrix(y_test, y_pred)
        print(matrice_confusion)
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=nom_classes, yticklabels=nom_classes, 
            annot=True, fmt ="d")
        plt.show()
            
with open("resultat/DonneesClassifieur2.json", "a") as f:
    f.write("]")
    f.close()


# In[193]:


from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)

##Comme nous savons que le 1 c'est spam et le 0 c'est ham, on peut écrire ceci
nom_classes = ["1940","1950","1960","1970","1980","1990","2000","2010","2020"]
report = classification_report(y_test, y_pred, target_names=nom_classes, digits=4)
print(report)


# In[194]:


## On peut enregistrer le classification report pour s'en servir plus tard
with open("resultat/report_classifier=perceptron_dataset=date.txt", "w") as w:
    w.write(report)


# In[195]:


with open("resultat/report_classifier=perceptron_dataset=date.txt") as f:
    r = f.read()
print(r)


# In[196]:


import json
report = classification_report(y_test, y_pred, target_names=nom_classes, digits=4, output_dict = True)
print(report)
with open("resultat/classificatio_report.json", "w") as w:
    w.write(json.dumps(report, indent=2))


# In[198]:


from sklearn.metrics import confusion_matrix

matrice_confusion = confusion_matrix(y_test, y_pred)
print(matrice_confusion)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(matrice_confusion, cmap = plt.cm.Reds, 
            xticklabels=nom_classes, yticklabels=nom_classes, 
            annot=True, fmt ="d")

#sns.heatmap(matrice_confusion, xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], cmap = plt.cm.Greys )
#plt.savefig()
plt.show()


# In[ ]:





# In[ ]:





# # Test sur un exemple de la structure du fichier .json

# In[128]:


x = {"555" :{
    "blabla" : 3,
    "toto" : 5,
    "tutu" : 4,
    "info_titre": [
      {
        "Titre": "2\u00e8me Lettre \u00e0 Toussenot by\u00a0Georges\u00a0Brassens",
        "Auteur": "Georges Brassens"
      },
      {
        "Titre": "Tout le long des rues by\u00a0Tino\u00a0Rossi",
        "Auteur": "Tino Rossi"
      }
    ],
    "chanson": [
      "Paris, 24.10.1946\n\nCher ami,\n\nLa semaine prochaine, nous essayerons de publier ton \u00e9tude sur le style, et cela me co\u00fbtera une formidable engueulade de la part du Comit\u00e9 national (car il existe un Comit\u00e9 national !) qui est assez r\u00e9fractaire aux choses du cin\u00e9ma, ainsi qu'a celles de l'Esprit d'ailleurs... \u00c0 tel point que, pour \u00e9viter de tout envoyer promener par suite de son insistance \u00e0 me consid\u00e9rer comme le dernier des imb\u00e9ciles, j'ai d\u00fb cesser d'\u00e9crire mes articles hebdomadaires. Cela me permet de me consacrer \u00e0 mes po\u00e9sies et \u00e0 ma pipe. Il para\u00eet que les lecteurs du Libertaire ne prennent aucune esp\u00e8ce d'int\u00e9r\u00eat \u00e0 la lecture de mes << conneries >> ! C'est possible, apr\u00e8s tout. Excuse-moi de te raconter tout \u00e7a qui \u00e9videmment ne te concerne pas, mais il y a des moments o\u00f9, ext\u00e9nu\u00e9 par la stupidit\u00e9 bourbeuse de nu\u00e9es de cuistres opini\u00e2tres, on est oblig\u00e9 de faire appel \u00e0 des hommes dont les facult\u00e9s intellectuelles ne sont pas en froid avec la subtilit\u00e9.\nJ'esp\u00e8re que le financier qui se dispose - le malheureux ! - \u00e0 placer des capitaux dans notre prochain journal ne tardera pas \u00e0 nous envoyer son ch\u00e8que, car j'ai une violente envie d'\u00e9crire dans une feuille libre, entour\u00e9 d'\u00e9crivains et non de rustres et d'ignorants ab\u00e9c\u00e9daires. Esp\u00e9rons donc, mon vieux, et \u00e0 bient\u00f4t de te lire.\n\nAmicalement.\n\nGeorges Brassens.",
      "[Refrain]\nTout le long, le long des rues\nJe m'en vais la nuit venue\nChercher jusqu'au petit jour\nTout ce qui reste de notre amour\nPas un coin, pas un quartier\nQui ne parle tout entier\nDe mon r\u00eave disparu\nTout le long, le long des rues\n\n[Couplet unique]\nLe vent tra\u00eene sa rengaine\nDans un ciel aussi lourd que ma peine\nLes nuages sont l'image\nDes beaux jours qui ne reviendront plus\nDans la foule qui s'\u00e9coule\nLe pass\u00e9 lentement se d\u00e9roule\nJusqu'au tendre soir, o\u00f9 pleins d'espoir\nNos c\u0153urs se sont connus\n\n[Refrain]\nTout le long, le long des rues\nJe m'en vais la nuit venue\nChercher jusqu'au petit jour\nTout ce qui reste de notre amour\nVagabond sentimental\nMon c\u0153ur dans ce dernier bal\nPour une ombre disparue\nTout le long, le long des rues"
    ]
  },
     "556" :{
    "blabla" : 3,
    "toto" : 5,
    "tutu" : 4,
    "info_titre": [
      {
        "Titre": "2\u00e8me Lettre \u00e0 Toussenot by\u00a0Georges\u00a0Brassens",
        "Auteur": "Georges Brassens"
      },
      {
        "Titre": "Tout le long des rues by\u00a0Tino\u00a0Rossi",
        "Auteur": "Tino Rossi"
      }
    ],
    "chanson": [
      "Paris, 24.10.1946\n\nCher ami,\n\nLa semaine prochaine, nous essayerons de publier ton \u00e9tude sur le style, et cela me co\u00fbtera une formidable engueulade de la part du Comit\u00e9 national (car il existe un Comit\u00e9 national !) qui est assez r\u00e9fractaire aux choses du cin\u00e9ma, ainsi qu'a celles de l'Esprit d'ailleurs... \u00c0 tel point que, pour \u00e9viter de tout envoyer promener par suite de son insistance \u00e0 me consid\u00e9rer comme le dernier des imb\u00e9ciles, j'ai d\u00fb cesser d'\u00e9crire mes articles hebdomadaires. Cela me permet de me consacrer \u00e0 mes po\u00e9sies et \u00e0 ma pipe. Il para\u00eet que les lecteurs du Libertaire ne prennent aucune esp\u00e8ce d'int\u00e9r\u00eat \u00e0 la lecture de mes << conneries >> ! C'est possible, apr\u00e8s tout. Excuse-moi de te raconter tout \u00e7a qui \u00e9videmment ne te concerne pas, mais il y a des moments o\u00f9, ext\u00e9nu\u00e9 par la stupidit\u00e9 bourbeuse de nu\u00e9es de cuistres opini\u00e2tres, on est oblig\u00e9 de faire appel \u00e0 des hommes dont les facult\u00e9s intellectuelles ne sont pas en froid avec la subtilit\u00e9.\nJ'esp\u00e8re que le financier qui se dispose - le malheureux ! - \u00e0 placer des capitaux dans notre prochain journal ne tardera pas \u00e0 nous envoyer son ch\u00e8que, car j'ai une violente envie d'\u00e9crire dans une feuille libre, entour\u00e9 d'\u00e9crivains et non de rustres et d'ignorants ab\u00e9c\u00e9daires. Esp\u00e9rons donc, mon vieux, et \u00e0 bient\u00f4t de te lire.\n\nAmicalement.\n\nGeorges Brassens.",
      "[Refrain]\nTout le long, le long des rues\nJe m'en vais la nuit venue\nChercher jusqu'au petit jour\nTout ce qui reste de notre amour\nPas un coin, pas un quartier\nQui ne parle tout entier\nDe mon r\u00eave disparu\nTout le long, le long des rues\n\n[Couplet unique]\nLe vent tra\u00eene sa rengaine\nDans un ciel aussi lourd que ma peine\nLes nuages sont l'image\nDes beaux jours qui ne reviendront plus\nDans la foule qui s'\u00e9coule\nLe pass\u00e9 lentement se d\u00e9roule\nJusqu'au tendre soir, o\u00f9 pleins d'espoir\nNos c\u0153urs se sont connus\n\n[Refrain]\nTout le long, le long des rues\nJe m'en vais la nuit venue\nChercher jusqu'au petit jour\nTout ce qui reste de notre amour\nVagabond sentimental\nMon c\u0153ur dans ce dernier bal\nPour une ombre disparue\nTout le long, le long des rues"
    ]
  },
}


# In[152]:


liste_date = []
liste_parole = []
for date, dic in x.items():
    chanson = dic.get("chanson")
    for z in chanson:
        liste_date.append(date)
        liste_parole.append(z)
print(liste_date)
print(liste_parole)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




