# RAG CV Matcher - Système Intelligent de Matching CV-Offres
##" 500+ CVs processed | LangChain + FAISS + Mistral"
Un système **RAG (Retrieval-Augmented Generation)** complet et en production conçu pour analyser automatiquement des CVs et les matcher avec des offres d'emploi.

##  Objectif

Construire un système capable d’extraire, structurer et vectoriser les informations d’un CV, de calculer la similarité avec des offres d’emploi, et de générer des explications contextuelles via un LLM.

##  Fonctionnalités Principales

- Extraction et parsing de CVs et offres (PDF / Texte)
- Structuration automatique des données non structurées
- Vectorisation avec un modèle d’embedding multilingue (E5-base)
- Indexation et recherche sémantique avec **FAISS**
- Calcul de similarité cosinus entre CV et offres
- Génération d’explications contextuelles avec **Mistral** (via Ollama)
- Agent IA autonome utilisant **LangChain** + moteur de recherche web (DuckDuckGo) pour recommander des formations personnalisées
- API REST complète développée avec **Django**
- Traitements asynchrones avec **RabbitMQ**

##  Technologies Utilisées

- **Langage** : Python
- **Frameworks IA** : LangChain, RAG, FAISS
- **LLM** : Mistral (via Ollama)
- **Backend** : Django + REST API
- **Asynchrone** : RabbitMQ
- **Base de données** : PostgreSQL

##  Résultats Obtenus

- Traitement automatisé de **plus de 500 CVs**
- Génération d’explications contextuelles précises et pertinentes
- Système capable de recommander des formations adaptées au profil du candidat
- Architecture scalable et prête pour une utilisation en production



