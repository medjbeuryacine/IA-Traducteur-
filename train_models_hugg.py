import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate
import numpy as np
import os
from datetime import datetime

class TranslationTrainer:
    """Classe pour entraîner des modèles de traduction sur OPUS Books"""
    
    def __init__(self):
        self.dataset = None
        self.tokenized_datasets = None
        self.models_info = [
            {
                "name": "marian-finetuned",
                "base_model": "Helsinki-NLP/opus-mt-en-fr",
                "output_dir": "./models/marian-finetuned-opus",
                "description": "Marian MT model fine-tuned on OPUS Books"
            },
            {
                "name": "t5-finetuned", 
                "base_model": "google-t5/t5-small",
                "output_dir": "./models/t5-finetuned-opus",
                "description": "T5-small model fine-tuned for EN->FR translation"
            }
        ]
        
        # Métriques d'évaluation
        self.bleu_metric = evaluate.load("sacrebleu")
        self.rouge_metric = evaluate.load("rouge")
    
    def load_and_prepare_dataset(self, max_train_samples=50000, max_eval_samples=5000):
        """Charger et préparer le dataset OPUS Books"""
        print("📚 Chargement du dataset OPUS Books...")
        
        try:
            # Charger le dataset complet
            dataset = load_dataset("opus_books", "en-fr")
            
            # Utiliser le split train et créer train/validation
            train_dataset = dataset["train"]
            
            # Limiter la taille pour l'entraînement (optionnel)
            if len(train_dataset) > max_train_samples:
                train_dataset = train_dataset.select(range(max_train_samples))
            
            # Créer un split train/validation
            dataset_split = train_dataset.train_test_split(test_size=0.1, seed=42)
            
            # Limiter la validation si nécessaire
            eval_dataset = dataset_split["test"]
            if len(eval_dataset) > max_eval_samples:
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            
            self.dataset = {
                "train": dataset_split["train"],
                "validation": eval_dataset
            }
            
            print(f"✅ Dataset préparé:")
            print(f"   📈 Entraînement: {len(self.dataset['train']):,} exemples")
            print(f"   🧪 Validation: {len(self.dataset['validation']):,} exemples")
            
            # Afficher quelques exemples
            print(f"\n📖 Exemples du dataset:")
            for i in range(3):
                example = self.dataset["train"][i]
                print(f"   {i+1}. EN: {example['translation']['en'][:100]}...")
                print(f"      FR: {example['translation']['fr'][:100]}...")
                print()
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
        
        return True
    
    def prepare_data_for_model(self, model_info, max_length=128):
        """Préparer les données pour un modèle spécifique"""
        print(f"🔧 Préparation des données pour {model_info['name']}...")
        
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_info["base_model"])
        
        # Fonction de preprocessing différente selon le modèle
        if "t5" in model_info["base_model"].lower():
            # Pour T5, préfixer avec la tâche
            def preprocess_function(examples):
                inputs = [f"translate English to French: {ex['en']}" for ex in examples["translation"]]
                targets = [ex['fr'] for ex in examples["translation"]]
                model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
                
                # Tokeniser les targets
                labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
        else:
            # Pour Marian et autres modèles de traduction
            def preprocess_function(examples):
                inputs = [ex['en'] for ex in examples["translation"]]
                targets = [ex['fr'] for ex in examples["translation"]]
                model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
                
                # Tokeniser les targets
                labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
        
        # Appliquer le preprocessing
        tokenized_datasets = {
            "train": self.dataset["train"].map(
                preprocess_function,
                batched=True,
                remove_columns=self.dataset["train"].column_names
            ),
            "validation": self.dataset["validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=self.dataset["validation"].column_names
            )
        }
        
        return tokenizer, tokenized_datasets
    
    def compute_metrics(self, eval_preds, tokenizer):
        """Calculer les métriques BLEU et ROUGE"""
        preds, labels = eval_preds
        
        # Décoder les prédictions et labels
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Remplacer -100 par pad_token_id pour le décodage
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Décoder
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Nettoyer les textes
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Calculer BLEU
        bleu_result = self.bleu_metric.compute(
            predictions=decoded_preds, 
            references=[[label] for label in decoded_labels]
        )
        
        # Calculer ROUGE
        rouge_result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        
        return {
            "bleu": bleu_result["score"],
            "rouge1": rouge_result["rouge1"],
            "rouge2": rouge_result["rouge2"],
            "rougeL": rouge_result["rougeL"]
        }
    
    def train_model(self, model_info, num_epochs=3, batch_size=16, learning_rate=5e-5):
        """Entraîner un modèle spécifique"""
        print(f"\n🚀 Début de l'entraînement: {model_info['name']}")
        print(f"   📦 Modèle de base: {model_info['base_model']}")
        print(f"   📁 Sortie: {model_info['output_dir']}")
        
        # Préparer les données
        tokenizer, tokenized_datasets = self.prepare_data_for_model(model_info)
        
        # Charger le modèle
        model = AutoModelForSeq2SeqLM.from_pretrained(model_info["base_model"])
        
        # Créer le répertoire de sortie
        os.makedirs(model_info["output_dir"], exist_ok=True)
        
        # Configuration de l'entraînement
        training_args = Seq2SeqTrainingArguments(
            output_dir=model_info["output_dir"],
            eval_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_epochs,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),  # Utiliser FP16 si GPU disponible
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            push_to_hub=False,
            logging_dir=f"{model_info['output_dir']}/logs",
            logging_steps=100,
            save_strategy="epoch",
            report_to=None,  # Désactiver wandb/tensorboard
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # Fonction de métrique avec tokenizer
        def compute_metrics_wrapper(eval_preds):
            return self.compute_metrics(eval_preds, tokenizer)
        
        # Créer le trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_wrapper
        )
        
        # Entraîner le modèle
        print(f"🔥 Démarrage de l'entraînement...")
        train_result = trainer.train()
        
        # Sauvegarder le modèle final
        trainer.save_model()
        tokenizer.save_pretrained(model_info["output_dir"])
        
        # Évaluation finale
        print(f"📊 Évaluation finale...")
        eval_result = trainer.evaluate()
        
        # Sauvegarder les métriques
        results = {
            "model_name": model_info["name"],
            "base_model": model_info["base_model"],
            "training_time": train_result.metrics["train_runtime"],
            "final_metrics": eval_result,
            "output_dir": model_info["output_dir"]
        }
        
        # Sauvegarder les résultats
        import json
        with open(f"{model_info['output_dir']}/training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Entraînement terminé pour {model_info['name']}")
        print(f"   📊 BLEU final: {eval_result.get('eval_bleu', 0):.4f}")
        print(f"   📁 Modèle sauvegardé: {model_info['output_dir']}")
        
        return results
    
    def train_all_models(self):
        """Entraîner tous les modèles configurés"""
        if not self.dataset:
            print("❌ Dataset non chargé. Appelez load_and_prepare_dataset() d'abord.")
            return
        
        results = []
        start_time = datetime.now()
        
        print(f"🎯 Début de l'entraînement de {len(self.models_info)} modèles")
        print(f"⏰ Heure de début: {start_time.strftime('%H:%M:%S')}")
        
        for i, model_info in enumerate(self.models_info, 1):
            print(f"\n{'='*60}")
            print(f"📈 MODÈLE {i}/{len(self.models_info)}: {model_info['name']}")
            print(f"{'='*60}")
            
            try:
                result = self.train_model(model_info)
                results.append(result)
                print(f"✅ Succès: {model_info['name']}")
                
            except Exception as e:
                print(f"❌ Erreur lors de l'entraînement de {model_info['name']}: {e}")
                continue
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"🎉 ENTRAÎNEMENT TERMINÉ")
        print(f"{'='*60}")
        print(f"⏰ Temps total: {total_time}")
        print(f"✅ Modèles entraînés: {len(results)}/{len(self.models_info)}")
        
        # Résumé des résultats
        for result in results:
            print(f"\n📊 {result['model_name']}:")
            print(f"   📁 Chemin: {result['output_dir']}")
            print(f"   📊 BLEU: {result['final_metrics'].get('eval_bleu', 0):.4f}")
            print(f"   ⏱️ Temps: {result['training_time']:.1f}s")
        
        return results

def main():
    """Fonction principale pour lancer l'entraînement"""
    print("🚀 ENTRAÎNEMENT DE MODÈLES DE TRADUCTION")
    print("=" * 50)
    
    # Créer le trainer
    trainer = TranslationTrainer()
    
    # Charger le dataset
    if not trainer.load_and_prepare_dataset():
        print("❌ Échec du chargement du dataset")
        return
    
    # Entraîner tous les modèles
    results = trainer.train_all_models()
    
    print(f"\n🎯 Entraînement terminé!")
    print(f"📁 Modèles disponibles dans:")
    for model_info in trainer.models_info:
        print(f"   - {model_info['output_dir']}")
    
    print(f"\n💡 Pour utiliser ces modèles dans votre app.py:")
    print(f"   1. Copiez les chemins des modèles")
    print(f"   2. Modifiez la fonction setup_hf_models() dans app.py")
    print(f"   3. Remplacez les modèles par défaut par vos modèles entraînés")

if __name__ == "__main__":
    main()