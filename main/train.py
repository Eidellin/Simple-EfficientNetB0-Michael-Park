import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from Lion import Lion
    from data import train_loader, valid_loader
    from EfficientNetB0 import EfficientNetB0
except ImportError:
    from .Lion import Lion
    from .data import train_loader, valid_loader
    from .EfficientNetB0 import EfficientNetB0

def evaluate(model, valid_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_loss = running_loss / len(valid_loader)
    accuracy = correct_predictions / total_samples
    print(f"Validation Loss: {average_loss:.4f} - Validation Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Calculate precision and recall
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0


    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    # Calculate F1-score, handling the case where precision + recall is zero
    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    print(f'F1-score: {f1_score}')


    return accuracy

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, info, train=True):
    best_accuracy = 0
    best_accuracy_valid = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            print('.', end='', flush=True)

        average_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        print(f"\n --- Epoch {epoch + 1}/{num_epochs} - Train Loss: {average_loss:.4f} - Train Accuracy: {accuracy:.4f}")

        if train:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), f"../models/{info}_train.pth")

            valid_accuracy = evaluate(model, valid_loader, criterion)
            if valid_accuracy > best_accuracy_valid:
                best_accuracy_valid = valid_accuracy
                torch.save(model.state_dict(), f"../models/{info}_valid.pth")
            
# Evaluate the model prediction probabilities, and return the list of probabilities, class predictions and sample paths
def evaluate_proba(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    all_proba = []
    cls_predictions = []

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)

            # Determine the class probabilities and predictions of the inputs
            model_proba = F.softmax(outputs, dim=1)

            for proba in model_proba:
                cls_predict = torch.argmax(proba).item()
                input_proba = max(proba[0].item(), proba[1].item())
                all_proba.append(input_proba)
                cls_predictions.append(cls_predict)

    # The predicted class corresponds to the class of the largest predicted probability
    idx = np.argmax(all_proba)
    cls_prediction = cls_predictions[idx]

    return all_proba, cls_prediction

def run_training_sample():
    model = EfficientNetB0(num_classes=2, stochastic_depth_prob=0.2).to(device)
    
    num_epochs = 30
    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, 'EfficientNetB0_Lion')
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_training_sample()
    elif sys.argv[1] == 'test':
        model_path = "../models/EfficientNetB0_Lion_valid.pth"
        model = EfficientNetB0(num_classes=2, stochastic_depth_prob=0.2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        criterion = nn.CrossEntropyLoss()
        evaluate(model, valid_loader, criterion)
    else:
        print("Invalid argument. Please use 'test' to evaluate the model.")

"""
Local Machine (MacBook Pro (16-inch, 2019)):
Validation Loss: 0.6679 - Validation Accuracy: 0.7500
Confusion Matrix:
[[35  1]
 [17 19]]
Precision: 0.95
Recall: 0.5277777777777778
F1-score: 0.6785714285714285

T4 GPU (Google Colab):
Train Loss: 0.2131 - Train Accuracy: 0.9100
Validation Loss: 0.4930 - Validation Accuracy: 0.7639
Confusion Matrix:
[[30  6]
 [11 25]]
Precision: 0.8064516129032258
Recall: 0.6944444444444444
F1-score: 0.746268656716418
"""