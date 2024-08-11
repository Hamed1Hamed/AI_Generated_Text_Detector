
from itertools import product
from ClassifierDataset import ClassifierDataset
from transformers import AutoTokenizer
from Classifier import *
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import json
" This class is used to evaluate the model on the testing set."
class Evaluator:
    def __init__(self, model_path, model_name, num_labels, device):
        # Set up logging
        logging.basicConfig(filename='classifier.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ModelEvaluator...")

        # Determine device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")


        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        # Load model
        try:
            self.model = CustomModel(model_name, num_labels)
            self.model.load_state_dict(torch.load(model_path))
            self.device = device
            self.model.to(self.device)
            self.loss_fn = torch.nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for binary classification
            self.logger.info("Model loaded for evaluation. The model is now evaluating the testing set using existing weights.")
            self.logger.info(f"Model loaded for evaluation. Loaded best model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model. Error: {e}")
            raise

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        y_scores = []

        progress_bar = tqdm(data_loader, desc="Evaluating (testing set)", leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    inputs, labels = batch
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    labels = labels.to(self.device)

                    logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).squeeze(
                        -1)
                    loss = self.loss_fn(logits, labels.float())
                    total_loss += loss.item()

                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > 0.5).long()
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())
                    y_scores.extend(probabilities.cpu().numpy())

                except Exception as e:
                    self.logger.error("Error during model evaluation: {}".format(e), exc_info=True)

            avg_loss = total_loss / len(data_loader)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc_score = roc_auc_score(y_true, y_scores)

            # Logging each metric on a new line
            self.logger.info("Testing Evaluation Metrics:")
            self.logger.info(f"  - Average Loss: {avg_loss}")
            self.logger.info(f"  - Accuracy: {accuracy}")
            self.logger.info(f"  - Precision: {precision}")
            self.logger.info(f"  - Recall: {recall}")
            self.logger.info(f"  - F1 Score: {f1}")
            self.logger.info(f"  - AUC-ROC: {auc_score}")

            # Log confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            self.logger.info(f"  - Confusion Matrix: \n{cm}")

            # Optionally plot results if needed
            self.plot_roc_curve(y_true, y_scores, auc_score)
            self.plot_confusion_matrix(y_true, y_pred)
            self.plot_normalized_confusion_matrix(y_true, y_pred)

            return avg_loss

    def plot_roc_curve(self,y_true, y_scores, auc_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve.png')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        classes = ['AI-generated', 'Human-written']
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0

        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title(f'Testing Confusion Matrix\nSensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.tight_layout()
        plt.show()

    def plot_normalized_confusion_matrix(self, y_true, y_pred):
        classes = ['AI-generated', 'Human-written']
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

        plt.figure(figsize=(6, 6))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.PuBuGn)
        plt.title('Normalized Testing Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        fmt = '.2f'  # Change format from integer to float for normalized matrix
        thresh = cm_normalized.max() / 2.
        for i, j in product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
            plt.text(j, i, format(cm_normalized[i, j], fmt), horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()


# Function to run the evaluation independently
def run_evaluation():

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = ClassifierDataset(tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config['testing_batch_size'], shuffle=False)

    # Initialize Model Evaluator
    best_model_path = os.path.join(config['checkpoint_path'], "best_model.pt")
    model_evaluator = Evaluator(best_model_path, model_name, num_labels=2, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run evaluation
    model_evaluator.evaluate(test_loader)

if __name__ == '__main__':
    run_evaluation()
