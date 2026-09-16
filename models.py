import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.datasets import make_classification

class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        # Simple CNN architecture for demonstration
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 16 * 16, 1) # Example for 32x32 input
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.sigmoid(self.fc(x))
        return x

def get_xgboost_model():
    """Returns a pre-trained XGBoost model for demonstration."""
    # Generate dummy clinical dataset to initialize the XGBClassifier
    # Features (in order from app.py): 
    # diameter, spiculation, lobulation, pack_years, age, calcification
    X, y = make_classification(n_samples=200, n_features=6, random_state=42)
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X, y)
    return model
