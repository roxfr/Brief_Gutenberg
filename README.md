# Développement et Déploiement d'un Agent LLM sur Azure

## Contexte du Projet

Ce projet a pour objectif de développer un agent basé sur un modèle de langage (LLM) capable de répondre à des questions sur des livres provenant de [Project Gutenberg](https://gutenberg.org/). L'agent doit être capable de traiter des requêtes variées, comme extraire des informations sur les livres et leurs personnages, ainsi que d'interagir avec le texte complet d'un livre.

## Objectifs

1. **Récupérer les données** : Scrapper au moins 5000 livres sur Project Gutenberg, en se concentrant sur la section "about" de chaque livre.
2. **Construire un agent avec LangChain** : L'agent doit pouvoir :
   - Répondre à des questions sur les livres et leurs attributs.
   - Récupérer une liste des noms des personnages mentionnés dans le résumé du livre.
   - Accéder au texte complet d'un livre et répondre à des questions à ce sujet.
3. **Évaluer l'agent** : Utiliser des techniques de RAG (Retrieval-Augmented Generation) pour améliorer les performances de l'agent.
4. **Construire une API avec FastAPI** : Rendre l'agent accessible via une API.
5. **(Facultatif) Déployer l'application sur Azure** : Rendre l'API accessible sur le cloud.

## Technologies Utilisées

- **Langages** : Python
- **Frameworks** : FastAPI, LangChain
- **Cloud** : Azure
- **Base de données** : Chroma Vector Store