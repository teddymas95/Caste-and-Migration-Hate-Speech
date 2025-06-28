import os
import random
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_curve, auc
from transformers import BertTokenizer, BertForSequenceClassification, XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from tqdm import tqdm
from nlpaug.augmenter.word import SynonymAug
import nltk
from itertools import cycle

# Download necessary NLTK data (run only once)
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HateSpeechDataset(Dataset):
    """PyTorch Dataset for Hate Speech Detection."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CNNClassifier(nn.Module):
    """Simple CNN for text classification."""
    def __init__(self, vocab_size, embedding_dim=300, num_filters=100, filter_sizes=[3, 4, 5], num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids).unsqueeze(1)  # [batch, 1, seq_len, embed_dim]
        convs = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, alpha=0.9, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

class HateSpeechDetector:
    """Main class for Hate Speech Detection Pipeline."""
    def __init__(self, model_type='xlm-roberta-large', num_labels=2, label_column='label'):
        self.model_type = model_type
        self.label_column = label_column
        self.num_labels = num_labels
        self.best_f1 = 0.0
        self.patience = 3
        self.patience_counter = 0
        self.augmenter = SynonymAug(aug_p=0.5)

        if model_type == 'mbert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
        elif model_type == 'xlm-roberta-base':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            self.model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=num_labels)
        elif model_type == 'xlm-roberta-large':
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
            self.model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large', num_labels=num_labels)
        elif model_type == 'cnn':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            vocab_size = self.tokenizer.vocab_size
            self.model = CNNClassifier(vocab_size, num_classes=num_labels)
        else:
            raise ValueError("Invalid model type")
        self.model.to(device)

    def augment_text(self, text, label):
        """Performs random augmentation for positive class."""
        if label == 1 and random.random() > 0.3:
            text = self.augmenter.augment(text)[0]
        words = text.split()
        if len(words) > 3:
            if random.random() > 0.5:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            if label == 1 and random.random() > 0.6:
                words = [w for w in words if random.random() > 0.2]
        return ' '.join(words)

    def load_and_prepare_data(self, train_file, dev_file):
        """Loads data, applies oversampling & augmentation."""
        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)

        for df, name in [(train_df, 'train'), (dev_df, 'dev')]:
            for col in ['id', 'text', self.label_column]:
                if col not in df.columns:
                    raise KeyError(f"{name} data must contain '{col}'.")

        # Oversample class 1 to approx 4000 samples
        class_1_df = train_df[train_df[self.label_column] == 1]
        target_size = 4000
        if len(train_df) < target_size and len(class_1_df) > 0:
            oversample_factor = int((target_size - len(train_df)) / len(class_1_df)) + 1
            oversampled_class_1 = pd.concat([class_1_df] * oversample_factor, ignore_index=True)
            train_df = pd.concat([train_df, oversampled_class_1], ignore_index=True)
            train_df = train_df.sample(n=min(target_size, len(train_df)), random_state=42)

        logging.info(f"Train distribution after oversampling: {train_df[self.label_column].value_counts().to_dict()}")
        logging.info(f"Dev distribution: {dev_df[self.label_column].value_counts().to_dict()}")

        train_df['text'] = [self.augment_text(row['text'], row[self.label_column]) for _, row in train_df.iterrows()]
        return train_df, dev_df

    def create_dataloaders(self, train_df, dev_df, batch_size=16):
        """Create PyTorch DataLoaders."""
        train_dataset = HateSpeechDataset(train_df['text'].values, train_df[self.label_column].values, self.tokenizer)
        dev_dataset = HateSpeechDataset(dev_df['text'].values, dev_df[self.label_column].values, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, dev_loader

    def train_model(self, train_loader, dev_loader, epochs=3, learning_rate=2e-5, accumulation_steps=4):
        """Train the model with early stopping."""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        criterion = FocalLoss(alpha=0.9, gamma=3.0)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()

            for i, batch in enumerate(tqdm(train_loader, desc="Training")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ) if self.model_type != 'cnn' else self.model(input_ids, attention_mask)

                logits = outputs.logits if self.model_type != 'cnn' else outputs
                loss = criterion(logits, labels)
                total_loss += loss.item()
                (loss / accumulation_steps).backward()
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            f1, dev_loss, *_ = self.evaluate_model(dev_loader, save_best=True)
            print(f"Validation F1: {f1:.4f}, Validation loss: {dev_loss:.4f}")
            scheduler.step(f1)
            if f1 <= self.best_f1:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break
            else:
                self.patience_counter = 0

    def evaluate_model(self, dev_loader, save_best=False):
        """Evaluate model and print metrics."""
        self.model.eval()
        total_eval_loss = 0
        predictions, true_labels, probabilities = [], [], []
        criterion = FocalLoss(alpha=0.9, gamma=3.0)
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ) if self.model_type != 'cnn' else self.model(input_ids, attention_mask)
                logits = outputs.logits if self.model_type != 'cnn' else outputs
                loss = criterion(logits, labels)
                total_eval_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.where(probs[:, 1] > 0.3, 1, 0).cpu().numpy()
                labels = labels.cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels)
                probabilities.extend(probs[:, 1].cpu().numpy())
        avg_loss = total_eval_loss / len(dev_loader)
        f1 = f1_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        if save_best and f1 > self.best_f1:
            self.best_f1 = f1
            self.save_model(f'best_model_{self.model_type}')
            print(f"New best model saved with F1 score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
        return f1, avg_loss, predictions, true_labels, probabilities, recall

    def save_model(self, output_dir):
        """Save model weights and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        if self.model_type != 'cnn':
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        print(f"Model saved to {output_dir}")

def plot_confusion_matrices(model_results, model_names):
    """Plot confusion matrices for each model."""
    plt.figure(figsize=(15, 10))
    for i, (model_name, result) in enumerate(zip(model_names, model_results)):
        cm = confusion_matrix(result['true_labels'], result['predictions'])
        plt.subplot(2, 2, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def plot_roc_curves(model_results, model_names):
    """Plot ROC curves for each model."""
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink'])
    for i, (model_name, result, color) in enumerate(zip(model_names, model_results, colors)):
        fpr, tpr, _ = roc_curve(result['true_labels'], result['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()

def create_comparison_table(model_results, model_names):
    """Create a pandas DataFrame comparing models."""
    data = {
        'Model': model_names,
        'F1 Score': [result['f1'] for result in model_results],
        'Recall': [result['recall'] for result in model_results],
        'Validation Loss': [result['loss'] for result in model_results]
    }
    return pd.DataFrame(data)

def main():
    train_file = 'data/train.csv'
    dev_file = 'data/dev.csv'
    model_types = ['mbert', 'xlm-roberta-base', 'xlm-roberta-large', 'cnn']
    model_names = ['mBERT', 'XLM-RoBERTa-base', 'XLM-RoBERTa-large', 'CNN']
    model_results = []
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        detector = HateSpeechDetector(model_type=model_type, label_column='label')
        train_df, dev_df = detector.load_and_prepare_data(train_file, dev_file)
        batch_size = 4 if model_type in ['xlm-roberta-large', 'mbert'] else 16
        train_loader, dev_loader = detector.create_dataloaders(train_df, dev_df, batch_size=batch_size)
        detector.train_model(train_loader, dev_loader, epochs=3, learning_rate=5e-6, accumulation_steps=4)
        f1, loss, predictions, true_labels, probabilities, recall = detector.evaluate_model(dev_loader)
        model_results.append({
            'f1': f1,
            'loss': loss,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'recall': recall
        })
    plot_confusion_matrices(model_results, model_names)
    plot_roc_curves(model_results, model_names)
    comparison_table = create_comparison_table(model_results, model_names)
    print("\nModel Comparison Table:")
    print(comparison_table)
    comparison_table.to_csv('model_comparison.csv', index=False)

if __name__ == "__main__":
    main()
