import flwr as fl
import torch
from models.classifier import build_model
from federated.fedprox import compute_fedprox_proximal_term

class FlowerHospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: int, train_loader, val_loader, config: dict):
        self.hospital_id = hospital_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(config).to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["federated"].get("lr", 0.001))
        
        self.model.train()
        for epoch in range(self.config["federated"].get("local_epochs", 1)):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss += criterion(out, y).item() * len(y)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return float(loss / max(1, total)), total, {"accuracy": float(correct / max(1, total))}
