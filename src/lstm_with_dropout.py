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

# Also MinMax-scaling what will be target; but saving orig. min and
# omin, omax = data["net_migration"].min(), data["net_migration"].max()
omin, omax = -2, 2
data["net_migration"] = minmax_scale(data["net_migration"])

# Drop 2022 & 2023
data = data[data["year"] < 2022]

data = data.sort_values(["iso3", "year"]).reset_index(drop=True)


data["net_migration_tp1"] = data.groupby(by="iso3").shift(-1)["net_migration"]
data = data.dropna(subset="net_migration_tp1")

# Define the target variable
target = "net_migration_tp1"

# Ensure that all features are numeric
features = data.drop(["iso3", "year", target], axis=1).select_dtypes(include=[np.number]).columns.tolist()

def create_sequences(df: pd.DataFrame, seq_length: int) -> list:
    sequences = []
    for _, group in df.groupby("iso3"):
        for i in range(len(group) - seq_length):
            seq = group.iloc[i:i+seq_length][features].values
            target = group.iloc[i+seq_length]["net_migration_tp1"]
            sequences.append((seq.astype(np.float32), float(target)))
    return sequences

class MigrationDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Split data into train, validation, and test sets
data_train = data[data['set'] == 'train']
data_val = data[data['set'] == 'val']
data_test = data[data['set'] == 'test']

# Define sequence length
seq_length = 1

# Create sequences
train_seqs = create_sequences(data_train, seq_length)
val_seqs = create_sequences(data_val, seq_length)
test_seqs = create_sequences(data_test, seq_length)

# Create Datasets
train_set = MigrationDataset(train_seqs)
val_set = MigrationDataset(val_seqs)
test_set = MigrationDataset(test_seqs)

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.6)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
input_size = len(train_seqs[0][0][0])
hidden_size = 128
num_layers = 2
output_size = 1

model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

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

scale = omax - omin

print(f"Metrics rescaled to original units:\n\nMSE: {metrics['MSE'] * (scale ** 2)}\nRMSE: {metrics['RMSE'] * scale}\nMAE: {metrics['MAE'] * scale}")

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
plt.show()

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
plt.show()

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
plt.show()


"""
Original Matrics: {'MSE': 0.00010090353368598568, 'RMSE': 0.01004507509608493, 'MAE': 0.005092993445162263}
Metrics rescaled to original units:

MSE: 0.001614456538975771
RMSE: 0.04018030038433972
MAE: 0.02037197378064905
"""

# Combine the validation and test datasets
combined_data = pd.concat([data_val, data_test])

# Get the list of unique countries (iso3 codes) in the combined dataset
countries = combined_data["iso3"].unique()

# Dictionary to store the MSE results
mse_results = {"country": [], "mse_combined": []}


# Function to collect predictions
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

# Function to rescale data back to original units
def rescale(scaled: list, omin: float, omax: float) -> np.array:
    return [x * (omax - omin) + omin for x in scaled]

# Function to calculate MSE for a given country on the combined validation and test datasets
def calculate_mse_for_country_combined(country, data, seq_length, model):
    country_data = data[data["iso3"] == country]
    country_seqs = create_sequences(country_data, seq_length)
    country_set = MigrationDataset(country_seqs)
    country_loader = DataLoader(country_set, batch_size=32, shuffle=False)

    pred, act = collect_predictions(model, country_loader)
    pred = rescale(pred, omin, omax)
    act = rescale(act, omin, omax)
    
    mse = mean_squared_error(act, pred)
    return mse

# Loop through each country and calculate MSE on the combined validation and test datasets
for country in countries:
    # Combined set MSE
    mse_combined = calculate_mse_for_country_combined(country, combined_data, seq_length=1, model=model)
    
    # Store the results
    mse_results["country"].append(country)
    mse_results["mse_combined"].append(mse_combined)

# Convert the results to a DataFrame for easy analysis
mse_df = pd.DataFrame(mse_results)

# Sort the DataFrame by MSE
mse_df_sorted = mse_df.sort_values(by="mse_combined", ascending=False)

# Display the top 10 countries with the highest MSE
print("\nTop 10 countries with highest MSE on combined validation and test datasets:")
print(mse_df_sorted.head(10))

# Display the bottom 10 countries with the lowest MSE
print("\nBottom 10 countries with lowest MSE on combined validation and test datasets:")
print(mse_df_sorted.tail(10))