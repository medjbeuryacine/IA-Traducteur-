# 🌍 Machine Translation Demo avec LLMs + Gradio

Application de démonstration pour comparer différents modèles de traduction automatique anglais-français avec évaluation des performances.

## 📋 Vue d'ensemble

Cette application compare **6 modèles de traduction** avec **5 températures différentes** pour analyser les performances de traduction anglais-français :
- 🦙 **2 modèles Ollama** (locaux)
- 🤗 **3 modèles HuggingFace** (fine-tunés sur OPUS Books)
- 📊 **Évaluation automatique** avec métriques BLEU et ROUGE

## 🤖 Modèles Utilisés

### **Modèles Ollama (Locaux)**
- **Llama 3.2:3b** - Modèle rapide et léger pour traduction
- **Mistral 7b** - Modèle plus précis pour textes complexes

### **Modèles HuggingFace (Fine-tunés)**
- **Marian-Finetuned** - Basé sur `Helsinki-NLP/opus-mt-en-fr`
- **T5-Finetuned** - Basé sur `google-t5/t5-small`
- **mBART-Finetuned** - Basé sur `facebook/mbart-large-50-many-to-many-mmt`

## 📚 Dataset

**OPUS Books** - Collection de livres libres de droits alignés en anglais-français
- **45,000 exemples** d'entraînement
- **5,000 exemples** de validation
- Textes littéraires de qualité pour un entraînement robuste

## 🌡️ Températures Testées

- **0.0** - Déterministe (traductions cohérentes)
- **0.2** - Légèrement créatif
- **0.5** - Équilibré
- **0.8** - Plus créatif
- **1.0** - Très créatif (moins prévisible)

## 📊 Métriques d'Évaluation

### **BLEU Score**
Mesure la similarité entre traduction générée et référence
- **>40** : Excellent
- **20-40** : Bon
- **<20** : Faible

### **ROUGE Scores**
Mesure le recouvrement de n-grammes
- **ROUGE-1** : Mots uniques
- **ROUGE-2** : Bigrammes
- **ROUGE-L** : Plus longue sous-séquence commune

## 🚀 Installation et Usage

### **Prérequis**
```bash
# Cloner le repository
git clone <votre-repo>
cd machine-translation-demo

# Créer l'environnement virtuel
python -m venv translator_env
source translator_env/bin/activate  # Linux/Mac
# ou translator_env\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### **Installer Ollama**
1. Télécharger depuis [ollama.ai](https://ollama.ai)
2. Installer les modèles :
```bash
ollama pull llama3.2:3b
ollama pull mistral:7b
```

### **Entraîner les modèles HuggingFace**
```bash
python train_models_hugg.py
```

### **Lancer l'application**
```bash
python app.py
```

Accéder à l'interface : `http://localhost:7860`

## 📁 Structure du Projet

```
machine-translation-demo/
├── app.py                     # Application principale Gradio
├── train_models_hugg.py       # Script d'entraînement des modèles
├── requirements.txt           # Dépendances Python
├── models/                    # Modèles fine-tunés
│   ├── marian-finetuned-opus/
│   ├── t5-finetuned-opus/
│   └── mbart-finetuned-opus/
├── results/                   # Résultats d'évaluation
│   ├── evaluation_results.csv
│   ├── bleu_scores.png
│   └── bleu_heatmap.png
└── README.md
```

## 🎯 Fonctionnalités

### **Interface Gradio avec 4 onglets :**

1. **🔤 Traduction Interactive**
   - Test en temps réel des modèles
   - Sélection du type de modèle et température
   - Comparaison directe des résultats

2. **📊 Évaluation Complète**
   - Évaluation automatique sur 50 exemples du dataset
   - Calcul des scores BLEU et ROUGE
   - Barre de progression en temps réel

3. **📈 Résultats Détaillés**
   - Tableau comparatif des performances
   - Classement par métrique
   - Export des résultats

4. **ℹ️ À propos**
   - Documentation complète
   - Explication des modèles et métriques

## 📊 Résultats de Performance

### **Classement par Score BLEU**
1. **Marian-Finetuned** : 49.02 ⭐
2. **T5-Finetuned** : 34.27
3. **mBART-Finetuned** : [En cours d'évaluation]
4. **Mistral 7b** : 5-7
5. **Llama 3.2:3b** : 1-2

### **Observations**
- Les **modèles fine-tunés** surpassent largement les modèles Ollama
- **Marian** excelle en précision (BLEU élevé)
- **T5** excelle en recouvrement (ROUGE élevé)
- La **température** a peu d'impact sur les modèles fine-tunés

## 🛠️ Technologies Utilisées

- **Interface** : Gradio 3.35+
- **Modèles** : Transformers 4.30+, Ollama
- **Évaluation** : SacreBLEU, ROUGE Score
- **Données** : Datasets (HuggingFace)
- **Visualisation** : Matplotlib, Seaborn
- **Backend** : PyTorch

## 📈 Améliorations Futures

- [ ] Ajout de nouveaux modèles (NLLB, mT5)
- [ ] Support d'autres paires de langues
- [ ] Évaluation sur datasets additionnels
- [ ] Interface web déployée
- [ ] API REST pour intégration

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouveaux modèles
- Améliorer l'interface
- Ajouter des métriques d'évaluation

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🎯 Citation

Si vous utilisez ce projet dans vos recherches, veuillez citer :

```bibtex
@misc{machine-translation-demo,
  title={Machine Translation Demo with LLMs and Gradio},
  author={[Votre Nom]},
  year={2025},
  url={[URL de votre repository]}
}
```

---

**Développé avec ❤️ pour la communauté NLP**