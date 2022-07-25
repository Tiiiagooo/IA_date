#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


corpus = ouvrir_json("tmp.json")


# In[ ]:


dictionnaire_ia5 = {}
for nom_artiste ,liste_chansons in tqdm.tqdm(corpus.items()):
    for element in liste_chansons:
        date = element["release_date"]
        if date is None:
            date = "None"
        titre = element["full_title"]
        #try:
        texte = element["lyrics"]
        #except:
            #pass
        try:
            artiste = element["album"]["artist"]["name"]
        except:
            artiste = "pas de nom"
        dictionnaire_ia5.setdefault(date, {"nbre_de_texte" : 0, "nbre_de_mot" : 0, "nbre_de_caractere" : 0})
        dictionnaire_ia5[date].setdefault(titre, {"Auteur":"", "Parole":""})
        dictionnaire_ia5[date][titre]["Auteur"] = artiste
        dictionnaire_ia5[date][titre]["Parole"] = texte
        if texte != None:
            dictionnaire_ia5[date]["nbre_de_texte"] += 1
            mot_split = Splittxt3(texte)
            nbre_carac = 0
            for mot in mot_split:
                for caractere in mot:
                    nbre_carac += 1
            dictionnaire_ia5[date]["nbre_de_mot"] += len(mot_split)
            dictionnaire_ia5[date]["nbre_de_caractere"] += nbre_carac
        #print(len(mot_split))
        #break
    #break
        
        #dictionnaire_ia5[date] = { titre : {}}
print(dictionnaire_ia5)
ecrire_json("dictionnaire_ia5.json", dictionnaire_ia5)


# In[ ]:




