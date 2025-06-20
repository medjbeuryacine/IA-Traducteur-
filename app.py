import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import requests
import json
import numpy as np
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import time
import os

class TranslationEvaluator:
    """Classe pour gérer tous les modèles de traduction et l'évaluation"""
    
    def __init__(self):
        self.temperatures = [0.0, 0.2, 0.5, 0.8, 1.0]
        self.ollama_models = ["llama3.2:3b", "mistral:7b"]
        self.hf_models = []
        self.dataset = None
        self.test_data = None
        self.results = {}
        
    def setup_dataset(self, max_samples=1000):
        """Charger et préparer le dataset OPUS Books"""
        print("📚 Chargement du dataset OPUS Books...")
        try:
            # Charger le dataset
            dataset = load_dataset("opus_books", lang1="en", lang2="fr", split="train")
            
            # Prendre un échantillon pour les tests
            self.dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            # Préparer les données de test (100 premiers exemples)
            test_size = min(100, len(self.dataset))
            self.test_data = self.dataset.select(range(test_size))
            
            print(f"✅ Dataset chargé: {len(self.dataset)} exemples")
            print(f"🧪 Données de test: {test_size} exemples")
            
            # Afficher un exemple
            example = self.dataset[0]
            print(f"\n📖 Exemple:")
            print(f"🇬🇧 EN: {example['translation']['en']}")
            print(f"🇫🇷 FR: {example['translation']['fr']}")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du dataset: {e}")
            # Dataset de secours
            self.create_fallback_dataset()
    
    def create_fallback_dataset(self):
        """Créer un petit dataset de secours si OPUS Books ne fonctionne pas"""
        print("🔄 Création d'un dataset de secours...")
        fallback_data = [
            {"en": "Hello, how are you?", "fr": "Bonjour, comment allez-vous ?"},
            {"en": "I love programming.", "fr": "J'aime programmer."},
            {"en": "The weather is beautiful today.", "fr": "Le temps est magnifique aujourd'hui."},
            {"en": "Can you help me?", "fr": "Pouvez-vous m'aider ?"},
            {"en": "Thank you very much.", "fr": "Merci beaucoup."},
        ]
        
        class FallbackDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return {"translation": self.data[idx]}
            
            def select(self, indices):
                selected_data = [self.data[i] for i in indices if i < len(self.data)]
                return FallbackDataset(selected_data)
        
        self.dataset = FallbackDataset(fallback_data)
        self.test_data = self.dataset
        print(f"✅ Dataset de secours créé: {len(fallback_data)} exemples")
    
    def setup_hf_models(self):
        """Configurer les modèles HuggingFace (modèles entraînés)"""
        print("🤗 Configuration des modèles HuggingFace...")
        
        try:
            # Modèle 1: Marian fine-tuné sur OPUS Books
            model1_path = "./models/marian-finetuned-opus"
            if os.path.exists(model1_path):
                model1 = pipeline(
                    "translation_en_to_fr",
                    model=model1_path,
                    tokenizer=model1_path
                )
                model1_name = "Marian-Finetuned"
                print(f"✅ Modèle 1 chargé: {model1_name}")
            else:
                # Fallback vers le modèle pré-entraîné
                model1 = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
                model1_name = "Helsinki-OPUS"
                print(f"⚠️ Modèle entraîné non trouvé, utilisation du modèle original")
            
            # Modèle 2: T5 fine-tuné sur OPUS Books  
            model2_path = "./models/t5-finetuned-opus"
            if os.path.exists(model2_path):
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                tokenizer2 = AutoTokenizer.from_pretrained(model2_path)
                model2_raw = AutoModelForSeq2SeqLM.from_pretrained(model2_path)
                model2 = pipeline(
                    "text2text-generation",
                    model=model2_raw,
                    tokenizer=tokenizer2
                )
                model2_name = "T5-Finetuned"
                print(f"✅ Modèle 2 chargé: {model2_name}")
            else:
                # Fallback vers le modèle pré-entraîné
                model2_name = "google-t5/t5-small"
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
                model2_raw = AutoModelForSeq2SeqLM.from_pretrained(model2_name)
                model2 = pipeline("text2text-generation", model=model2_raw, tokenizer=tokenizer2)
                model2_name = "T5-Original"
                print(f"⚠️ Modèle entraîné non trouvé, utilisation du modèle original")
            
            self.hf_models = [
                {"name": model1_name, "model": model1, "type": "translation"},
                {"name": model2_name, "model": model2, "type": "text2text"}
            ]
            
            print("✅ Modèles HuggingFace configurés")
            
        except Exception as e:
            print(f"❌ Erreur configuration HF: {e}")
            self.hf_models = []
        
    
    def translate_with_ollama(self, text: str, model: str, temperature: float) -> str:
        """Traduire avec un modèle Ollama"""
        try:
            prompt = f"Translate this English text to French: '{text}'"
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'options': {
                        'temperature': temperature,
                        'num_predict': 100
                    },
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"Erreur Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Erreur connexion Ollama: {e}"
    
    def translate_with_hf(self, text: str, model_info: dict, temperature: float) -> str:
        """Traduire avec un modèle HuggingFace"""
        try:
            model = model_info["model"]
            model_type = model_info["type"]
            
            if model_type == "translation":
                # Pour les modèles de traduction directs
                result = model(text, max_length=100, temperature=temperature, do_sample=temperature > 0)
                return result[0]['translation_text']
            
            elif model_type == "text2text":
                # Pour T5, il faut préfixer la tâche
                prefixed_text = f"translate English to French: {text}"
                result = model(
                    prefixed_text,
                    max_length=100,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    num_return_sequences=1
                )
                return result[0]['generated_text']
                
        except Exception as e:
            return f"Erreur HF: {e}"
    
    def calculate_bleu_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculer les scores BLEU et ROUGE"""
        try:
            # BLEU Score
            bleu_score = corpus_bleu(predictions, [references]).score
            
            # ROUGE Scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                rouge_scores['rouge1'] += scores['rouge1'].fmeasure
                rouge_scores['rouge2'] += scores['rouge2'].fmeasure
                rouge_scores['rougeL'] += scores['rougeL'].fmeasure
            
            # Moyenne
            n = len(predictions)
            rouge_scores = {k: v/n for k, v in rouge_scores.items()}
            
            return {
                'bleu': bleu_score,
                'rouge1': rouge_scores['rouge1'],
                'rouge2': rouge_scores['rouge2'],
                'rougeL': rouge_scores['rougeL']
            }
            
        except Exception as e:
            print(f"Erreur calcul métriques: {e}")
            return {'bleu': 0, 'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    
    def evaluate_all_models(self, progress_callback=None):
        """Évaluer tous les modèles avec toutes les températures"""
        print("🔬 Début de l'évaluation complète...")
        
        if not self.test_data:
            print("❌ Pas de données de test disponibles")
            return
        
        # Préparer les données de référence
        references = []
        sources = []
        
        for i, example in enumerate(self.test_data):
            if i >= 50:  # Limiter à 50 exemples pour l'évaluation
                break
            translation_data = example['translation']
            sources.append(translation_data['en'])
            references.append(translation_data['fr'])
        
        total_combinations = (len(self.ollama_models) + len(self.hf_models)) * len(self.temperatures)
        current_step = 0
        
        # Évaluer les modèles Ollama
        for model_name in self.ollama_models:
            for temp in self.temperatures:
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_combinations, f"Ollama {model_name} (T={temp})")
                
                print(f"🔄 Évaluation: Ollama {model_name} (température={temp})")
                
                predictions = []
                for source in sources:
                    pred = self.translate_with_ollama(source, model_name, temp)
                    predictions.append(pred)
                    time.sleep(0.1)  # Pause pour éviter la surcharge
                
                scores = self.calculate_bleu_rouge(predictions, references)
                
                self.results[f"Ollama_{model_name}_{temp}"] = {
                    'model_type': 'Ollama',
                    'model_name': model_name,
                    'temperature': temp,
                    'predictions': predictions[:5],  # Garder quelques exemples
                    'scores': scores
                }
        
        # Évaluer les modèles HuggingFace
        for model_info in self.hf_models:
            for temp in self.temperatures:
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_combinations, f"HF {model_info['name']} (T={temp})")
                
                print(f"🔄 Évaluation: HF {model_info['name']} (température={temp})")
                
                predictions = []
                for source in sources:
                    pred = self.translate_with_hf(source, model_info, temp)
                    predictions.append(pred)
                
                scores = self.calculate_bleu_rouge(predictions, references)
                
                self.results[f"HF_{model_info['name']}_{temp}"] = {
                    'model_type': 'HuggingFace',
                    'model_name': model_info['name'],
                    'temperature': temp,
                    'predictions': predictions[:5],
                    'scores': scores
                }
        
        print("✅ Évaluation terminée!")
        self.save_results()
    
    def save_results(self):
        """Sauvegarder les résultats"""
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # Créer un DataFrame des résultats
        results_data = []
        for key, result in self.results.items():
            row = {
                'Model_Type': result['model_type'],
                'Model_Name': result['model_name'],
                'Temperature': result['temperature'],
                'BLEU': result['scores']['bleu'],
                'ROUGE-1': result['scores']['rouge1'],
                'ROUGE-2': result['scores']['rouge2'],
                'ROUGE-L': result['scores']['rougeL']
            }
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        df.to_csv('results/evaluation_results.csv', index=False)
        
        # Créer des graphiques
        self.create_visualizations(df)
        
        print("💾 Résultats sauvegardés dans results/")
    
    def create_visualizations(self, df):
        """Créer des visualisations des résultats"""
        plt.style.use('seaborn-v0_8')
        
        # Graphique 1: BLEU scores par température
        plt.figure(figsize=(12, 8))
        
        for model_type in df['Model_Type'].unique():
            for model_name in df[df['Model_Type'] == model_type]['Model_Name'].unique():
                data = df[(df['Model_Type'] == model_type) & (df['Model_Name'] == model_name)]
                plt.plot(data['Temperature'], data['BLEU'], 
                        marker='o', label=f"{model_type}_{model_name}")
        
        plt.xlabel('Température')
        plt.ylabel('Score BLEU')
        plt.title('Scores BLEU par Température et Modèle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/bleu_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graphique 2: Heatmap des scores
        pivot_bleu = df.pivot_table(values='BLEU', 
                                   index=['Model_Type', 'Model_Name'], 
                                   columns='Temperature')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_bleu, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Heatmap des Scores BLEU')
        plt.tight_layout()
        plt.savefig('results/bleu_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Graphiques créés dans results/")

# Initialiser l'évaluateur global
evaluator = TranslationEvaluator()

def setup_app():
    """Initialiser l'application"""
    evaluator.setup_dataset()
    evaluator.setup_hf_models()

def translate_single(text, model_type, model_name, temperature):
    """Traduire un texte unique"""
    if not text.strip():
        return "⚠️ Veuillez entrer un texte à traduire"
    
    try:
        if model_type == "Ollama":
            result = evaluator.translate_with_ollama(text, model_name, temperature)
        elif model_type == "HuggingFace":
            # Trouver le modèle HF correspondant
            hf_model = next((m for m in evaluator.hf_models if m['name'] == model_name), None)
            if hf_model:
                result = evaluator.translate_with_hf(text, hf_model, temperature)
            else:
                result = "❌ Modèle HuggingFace non trouvé"
        else:
            result = "❌ Type de modèle non reconnu"
        
        return result
        
    except Exception as e:
        return f"❌ Erreur: {e}"

def run_full_evaluation(progress=gr.Progress()):
    """Lancer l'évaluation complète avec barre de progression"""
    def progress_callback(current, total, description):
        progress((current, total), desc=description)
    
    evaluator.evaluate_all_models(progress_callback)
    
    # Retourner un résumé
    if evaluator.results:
        summary = "✅ Évaluation terminée!\n\n"
        summary += f"📊 {len(evaluator.results)} combinaisons évaluées\n"
        summary += "📁 Résultats sauvegardés dans results/\n"
        summary += "📈 Graphiques créés\n\n"
        
        # Meilleurs scores
        best_bleu = max(evaluator.results.values(), key=lambda x: x['scores']['bleu'])
        summary += f"🏆 Meilleur BLEU: {best_bleu['scores']['bleu']:.2f} "
        summary += f"({best_bleu['model_type']} {best_bleu['model_name']}, T={best_bleu['temperature']})"
        
        return summary
    else:
        return "❌ Aucun résultat d'évaluation"

def get_results_dataframe():
    """Retourner les résultats sous forme de DataFrame"""
    if not evaluator.results:
        return pd.DataFrame({"Message": ["Aucune évaluation effectuée"]})
    
    results_data = []
    for key, result in evaluator.results.items():
        row = {
            'Modèle': f"{result['model_type']}_{result['model_name']}",
            'Température': result['temperature'],
            'BLEU': round(result['scores']['bleu'], 3),
            'ROUGE-1': round(result['scores']['rouge1'], 3),
            'ROUGE-2': round(result['scores']['rouge2'], 3),
            'ROUGE-L': round(result['scores']['rougeL'], 3)
        }
        results_data.append(row)
    
    return pd.DataFrame(results_data)

# Interface Gradio
def create_gradio_interface():
    """Créer l'interface Gradio"""
    
    with gr.Blocks(title="🌍 Démo Traduction Multimodèle", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌍 Démo de Traduction Automatique Anglais → Français
        
        Cette application compare **4 modèles de traduction** avec **5 températures différentes**:
        - 🦙 **Ollama**: Llama3.2:3b, Mistral:7b  
        - 🤗 **HuggingFace**: Helsinki-OPUS, Google-T5
        - 🌡️ **Températures**: 0.0, 0.2, 0.5, 0.8, 1.0
        - 📊 **Métriques**: BLEU et ROUGE scores
        """)
        
        with gr.Tabs():
            # Onglet 1: Traduction interactive
            with gr.Tab("🔤 Traduction Interactive"):
                gr.Markdown("### Testez la traduction en temps réel")
                
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="📝 Texte en anglais",
                            placeholder="Entrez votre texte en anglais...",
                            lines=3
                        )
                        
                        model_type = gr.Radio(
                            choices=["Ollama", "HuggingFace"],
                            label="🤖 Type de modèle",
                            value="Ollama"
                        )
                        
                        model_name = gr.Dropdown(
                            choices=["llama3.2:3b", "mistral:7b"],
                            label="📋 Nom du modèle",
                            value="llama3.2:3b"
                        )
                        
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            label="🌡️ Température",
                            value=0.5
                        )
                        
                        translate_btn = gr.Button("🚀 Traduire", variant="primary")
                    
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="🇫🇷 Traduction en français",
                            lines=3,
                            interactive=False
                        )
                
                # Mise à jour dynamique des choix de modèles
                def update_model_choices(model_type):
                    if model_type == "Ollama":
                        return gr.Dropdown(choices=["llama3.2:3b", "mistral:7b"], value="llama3.2:3b")
                    else:
                        return gr.Dropdown(choices=["Marian-Finetuned", "T5-Finetuned"], value="Marian-Finetuned")
                
                model_type.change(update_model_choices, model_type, model_name)
                translate_btn.click(translate_single, [input_text, model_type, model_name, temperature], output_text)
            
            # Onglet 2: Évaluation complète
            with gr.Tab("📊 Évaluation Complète"):
                gr.Markdown("### Évaluation automatique sur le dataset OPUS Books")
                
                with gr.Row():
                    eval_btn = gr.Button("🔬 Lancer l'évaluation complète", variant="primary", size="lg")
                
                eval_output = gr.Textbox(
                    label="📋 Résultats de l'évaluation",
                    lines=10,
                    interactive=False
                )
                
                eval_btn.click(run_full_evaluation, outputs=eval_output)
            
            # Onglet 3: Résultats détaillés
            with gr.Tab("📈 Résultats Détaillés"):
                gr.Markdown("### Tableau comparatif des performances")
                
                refresh_btn = gr.Button("🔄 Actualiser les résultats")
                results_table = gr.Dataframe(
                    headers=["Modèle", "Température", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"],
                    label="📊 Tableau des scores"
                )
                
                refresh_btn.click(get_results_dataframe, outputs=results_table)
            
            # Onglet 4: À propos
            with gr.Tab("ℹ️ À propos"):
                gr.Markdown("""
                ### 📚 Dataset utilisé
                **OPUS Books**: Collection de livres libres de droits alignés en anglais-français
                
                ### 🤖 Modèles testés
                
                **Ollama (Local)**:
                - **Llama3.2:3b**: Modèle rapide et efficace
                - **Mistral:7b**: Modèle plus précis pour textes complexes
                
                **HuggingFace (Cloud)**:
                - **Helsinki-NLP/opus-mt-en-fr**: Spécialisé EN→FR
                - **Google T5-small**: Modèle généraliste text-to-text
                
                ### 🌡️ Températures
                - **0.0**: Déterministe, traductions cohérentes
                - **0.2**: Légèrement créatif
                - **0.5**: Équilibré
                - **0.8**: Plus créatif
                - **1.0**: Très créatif, moins prévisible
                
                ### 📊 Métriques d'évaluation
                - **BLEU**: Mesure la similarité avec la référence
                - **ROUGE-1/2/L**: Mesure le recouvrement de n-grammes
                
                ### 🛠️ Technologies utilisées
                - **Gradio**: Interface web interactive
                - **Transformers**: Modèles HuggingFace
                - **Ollama**: Modèles locaux
                - **SacreBLEU & ROUGE**: Métriques d'évaluation
                """)
        
        # Initialisation au démarrage
        demo.load(setup_app)
    
    return demo

if __name__ == "__main__":
    print("🚀 Démarrage de l'application de traduction...")
    demo = create_gradio_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )