
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, TensorDataset
#from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device}")

data = pd.read_csv("migration-flow-prediction/data/final/preprocessed_full.csv", index_col=[0])

omin, omax = -2, 2
# data["net_migration"] = minmax_scale(data["net_migration"])

data = data[data["year"] < 2022][["iso3", "year", "net_migration"]]
data = data.pivot(index='year', columns='iso3', values='net_migration')

def create_sequences(series: list, seq: int) -> tuple:
    sequences = []
    targets = []

    for i in range(len(series) - seq):
        sequence = series[i:i+seq]
        target = series[i+seq]

        sequences.append(sequence)
        targets.append(target)
    
    y_test = targets[-2:]
    X_test = sequences[-2:]
    y_val = targets[-4:-2]
    X_val = sequences[-4:-2]
    y_train = targets[:-4]
    X_train = sequences[:-4]

    return X_train, y_train, X_val, y_val, X_test, y_test

seq = 3

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

for country in data.columns:
    series = data[country].reset_index(drop=True)
    X_train_i, y_train_i, X_val_i, y_val_i, X_test_i, y_test_i = create_sequences(series=series, seq=seq)
    X_train.extend(X_train_i)
    y_train.extend(y_train_i)
    X_val.extend(X_val_i)
    y_val.extend(y_val_i)
    X_test.extend(X_test_i)
    y_test.extend(y_test_i)

# This part makes me hate Python because above it implicitly creates data structures
# that don't really work here (lists of arrays and of series):

X_train = torch.tensor([x.values for x in X_train], dtype=torch.float32)
y_train = torch.tensor([x.item() for x in y_train], dtype=torch.float32)
X_val = torch.tensor([x.values for x in X_val], dtype=torch.float32)
y_val = torch.tensor([x.item() for x in y_val], dtype=torch.float32)
X_test = torch.tensor([x.values for x in X_test], dtype=torch.float32)
y_test = torch.tensor([x.item() for x in y_test], dtype=torch.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]
    
train_loader = DataLoader(TimeSeriesDataset(X_train.unsqueeze(-1), y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TimeSeriesDataset(X_val.unsqueeze(-1), y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset(X_test.unsqueeze(-1), y_test), batch_size=32, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get the output from the last time step
        out = self.fc1(out)
        out = F.relu(out)  # Add activation function
        out = self.dropout(out)
        out = self.fc2(out)
        return out



    
input_size = 1  # Number of features per time step; just use the first sequence to check...
hidden_size = 256
hidden_size2 = 128
num_layers = 3
output_size = 1  # Predicting a single value (migration flow) for the next year

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, hidden_size2=hidden_size2, num_layers=num_layers, output_size=output_size)
# model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def checkpoint(model) -> None:
    torch.save(model.state_dict(), "checkpoint.pth")

def restore_best_weights(model) -> None:
    model.load_state_dict(torch.load("checkpoint.pth"))

class EarlyStopping:
    
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("Inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            

stopper = EarlyStopping()

def validate(model: LSTMModel, val_loader: DataLoader, criterion: nn.HuberLoss) -> float:
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for features, targets in val_loader:
            pred = model(features)
            pred = pred.view(-1, 1)       
            targets = targets.view(-1, 1)  
            loss = criterion(pred, targets)
            total_loss += loss.item()
    
    return total_loss

def train(model: LSTMModel, train_loader: DataLoader, val_loader: DataLoader, epochs: int, criterion: nn.HuberLoss) -> list:

    losses = []

    for epoch in range(epochs):

        # Train step

        model.train()

        epoch_train_loss = 0

        for features, targets in train_loader:
            pred = model(features)
            pred = pred.view(-1, 1)       
            targets = targets.view(-1, 1) 
            loss = criterion(pred, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            
        # Validation step

        epoch_val_loss = validate(model, val_loader, criterion)

        stopper(epoch_val_loss)

        print(f"Epoch {epoch+1} done: Training loss: {epoch_train_loss:.4f}, Validation loss: {epoch_val_loss:.4f}, Patience counter: {stopper.counter}")
        losses.append({"epoch": epoch+1, "training": epoch_train_loss, "validation": epoch_val_loss})
        
        if stopper.counter == 0:
            checkpoint(model)
        
        if stopper.early_stop:
            print(f"Patience exceeded, stopping training & restoring best weights...")
            restore_best_weights(model)
            break
        


    return losses

losses = train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=100, criterion=criterion)

losses = pd.DataFrame(losses)

sns.lineplot(
    data=losses.melt(id_vars="epoch", var_name="loss"),
    x="epoch",
    y="value",
    hue="loss"
)

def eval(model: LSTMModel, test_loader: DataLoader):
    model.eval()
    
    mse = []
    mae = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = model(features)
            targets = targets.unsqueeze(1)
            
            mse_i = F.mse_loss(outputs, targets, reduction='mean').item()
            mse.append(mse_i)

            mae_i = F.l1_loss(outputs, targets, reduction='mean').item()
            mae.append(mae_i)
    
    metrics = {
        "MSE": np.mean(mse),
        "RMSE": np.sqrt(np.mean(mse)),
        "MAE": np.mean(mae)
    }
    
    return metrics

metrics = eval(model, test_loader)
metrics

print(f"Original Matrics: {metrics}")

"""scale = omax - omin

print(f"Metrics rescaled to original units:\n\nMSE: {metrics['MSE'] * (scale ** 2)}\nRMSE: {metrics['RMSE'] * scale}\nMAE: {metrics['MAE'] * scale}")
"""
def collect_predictions(model: LSTMModel, data_loader: DataLoader) -> tuple:
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
    
    return predictions, actuals

# Rescaling back to original units (migrants):

def rescale(scaled: list, omin: float, omax: float) -> np.array:
    return [x * (omax - omin) + omin for x in scaled]

predictions, actuals = collect_predictions(model, test_loader)
predictions, actuals = rescale(predictions, omin=omin, omax=omax), rescale(actuals, omin=omin, omax=omax)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=actuals, y=predictions, color="black", s=10)
# Main diagonal:
min_val = min(min(actuals), min(predictions))
max_val = max(max(actuals), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Net Migration\n(Test Set)')
# plt.show()

train_pred, train_act = collect_predictions(model, train_loader)
val_pred, val_act = collect_predictions(model, val_loader)

train_pred, train_act = rescale(train_pred, omin, omax), rescale(train_act, omin, omax)
val_pred, val_act = rescale(val_pred, omin, omax), rescale(val_act, omin, omax)

all_predictions = pd.concat([
    pd.DataFrame({"predicted": train_pred, "actual": train_act, "set": "training"}),
    pd.DataFrame({"predicted": val_pred, "actual": val_act, "set": "validation"}),
    pd.DataFrame({"predicted": predictions, "actual": actuals, "set": "testing"})
])

all_predictions['set'] = pd.Categorical(all_predictions['set'], categories=['testing', 'training', 'validation'])
all_predictions = all_predictions.sort_values('set').reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=all_predictions, 
    x="actual",
    y="predicted",
    hue="set",
    palette={"testing": "red", "training": "blue", "validation": "green"},
    s=20, 
    alpha=0.6
)
# Main diagonal:
min_val = min(min(all_predictions["actual"]), min(all_predictions["predicted"]))
max_val = max(max(all_predictions["actual"]), max(all_predictions["predicted"]))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Net Migration')
# plt.show()

all_predictions["error"] = all_predictions["predicted"] - all_predictions["actual"]
ax = sns.scatterplot(
    data=all_predictions,
    x="predicted",
    y="error",
    hue="set",
    palette={"testing": "red", "training": "blue", "validation": "green"},
    s=20, 
    alpha=0.5
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.axhline(y=0, color="black")
plt.xlabel('Predicted Values')
plt.ylabel('Residual')
plt.title('Residuals')
# plt.show()

"""
Original Matrics: {'MSE': 0.00038581134205222564, 'RMSE': 0.01964208089923839, 'MAE': 0.008084576310856002}
"""


