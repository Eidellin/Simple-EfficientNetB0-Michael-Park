import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, valid_loader):
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

    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1 Score: {f1:.4f}")
   
    return accuracy

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, info):
    best_accuracy = 0
    best_accuracy_valid = 0
    best_accuracy_train_for_valid = 0

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

        # Find the best train accuracy and save the model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"./models/{info}_train.pth")

        # Find the best validation accuracy and save the model
        valid_accuracy = evaluate(model, valid_loader)
        if valid_accuracy >= best_accuracy_valid:
            if valid_accuracy > best_accuracy_valid:
                best_accuracy_valid = valid_accuracy
            elif accuracy > best_accuracy_train_for_valid:
                pass
            else:
                continue
            best_accuracy_train_for_valid = accuracy
            torch.save(model.state_dict(), f"./models/{info}_valid.pth")
    
    return best_accuracy, best_accuracy_valid, best_accuracy_train_for_valid
            
if __name__ == "__main__":
    import sys
    import torch.nn as nn
    
    from EfficientNetB0 import EfficientNetB0
    from Lion import Lion
    from data import load_data_for_Penguins_vs_Turtles
    
    train_loader, valid_loader = load_data_for_Penguins_vs_Turtles()
    
    model = EfficientNetB0().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Lion(model.parameters(), lr=0.001)
    best_accuracy, best_accuracy_valid, best_accuracy_train_for_valid = train_model(model, train_loader, valid_loader, criterion, optimizer, 40, 'EfficientNetB0_Lion')
    print(f"Best Train Accuracy: {best_accuracy:.4f} - Best Validation Accuracy: {best_accuracy_valid:.4f} - Best Train Accuracy for Best Validation Accuracy Model: {best_accuracy_train_for_valid:.4f}")