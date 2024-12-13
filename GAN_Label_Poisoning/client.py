import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from multiprocessing import Process, Manager
from time import sleep
import argparse
import os
import pickle
import flwr as fl
from flwr.client import NumPyClient
import logging
logging.getLogger("flwr").setLevel(logging.ERROR)

if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float64
else:
    DEVICE = "cpu"
    DTYPE = torch.float64

ROUND = 1
ROUND_MAX = 5
SEED = 7
NUM_CLIENTS = 6
TRAIN_EPOCHS = 2

# Argument parser
parser = argparse.ArgumentParser(description="Flower Client for Federated Learning")
parser.add_argument(
    "--mode",
    type=str,
    choices=["legit", "attack"],
    required=True,
    help="Specify client mode: 'legit' or 'attack'",
)
parser.add_argument(
    "--type",
    type=str,
    choices=["gradient_flip", "label_flip", 'gan'],
    required=False,
    help="Specify attack type: 'gradient_flip', 'label_flip', or 'gan'",
)
args = parser.parse_args()
mode = args.mode
type = args.type
if mode=="attack" and args.type is None:
    parser.error("\"attack\" mode requires --type")

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(256, 512, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(512, 1024, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(1024, output_dim, dtype=DTYPE),
            nn.Tanh(), 
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x).view(-1, 50, 3).to(DTYPE)

class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(50, 32, kernel_size=3, dtype=DTYPE)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 50, dtype=DTYPE)
        self.fc2 = nn.Linear(50, 3, dtype=DTYPE)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 50, 512, dtype=DTYPE),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256, dtype=DTYPE),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1, dtype=DTYPE),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

class FlowerClient(NumPyClient):
    def __init__(self, client_id, mode, shared_dict, malicious):
        super().__init__()
        torch.manual_seed(SEED)
        self.net = CNN1DModel().to(DEVICE)
        self.discriminator = Discriminator().to(DEVICE)        
        self.generator = Generator(input_dim=100, output_dim=50*3).to(DEVICE)
        self.client_id = client_id
        self.trainloader, self.testloader = self.load_data()
        self.mode = mode
        self.shared_dict = shared_dict
        self.malicious = malicious
        self.criterion_d = nn.CrossEntropyLoss()  # Discriminator loss
        self.criterion_g = nn.CrossEntropyLoss()  # Generator loss
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)
        # For GAN poisoning
        self.index = 0
        self.fake_data, self.fake_labels = None, None
        self.fake_batch_size = 0
        self.fake_data_batches, self.fake_labels_batches = None, None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.as_tensor(v, dtype=DTYPE) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_number = config["server_round"]
        self.train(round_number)
        self.save_model_weights(round_number)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test(self.testloader)
        print(f"Client {self.client_id} Test Loss: {loss}, Test Accuracy: {accuracy}")
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}

    def save_model_weights(self, round_number):
        filename = f"model_r{round_number}_s{SEED}.pt"
        path = f"weights/{mode}/{filename}"
        os.makedirs(f"weights/{mode}", exist_ok=True)
        torch.save(self.get_parameters(None), path)
        print(f"Saved model weights at {path}")
    
    def save_gan_parameters(self):
        generator_path = f"weights/{self.mode}/generator_{self.client_id}.pt"
        discriminator_path = f"weights/{self.mode}/discriminator_{self.client_id}.pt"
        os.makedirs(f"weights/{self.mode}", exist_ok=True)
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        print(f"Saved generator and discriminator parameters for Client {self.client_id}")

    def load_gan_parameters(self):
        generator_path = f"weights/{self.mode}/generator_{self.client_id}.pt"
        discriminator_path = f"weights/{self.mode}/discriminator_{self.client_id}.pt"
        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path))
            print(f"Loaded generator parameters for Client {self.client_id}")
        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path))
            print(f"Loaded discriminator parameters for Client {self.client_id}")

    def load_data(self):
        with open(f"Dataset/node_{self.client_id}.pkl", "rb") as f:
            X_train, y_train, X_test, y_test, _, _ = pickle.load(f)
        X_train_tensor = torch.as_tensor(X_train, dtype=DTYPE).to(DEVICE)
        y_train_tensor = torch.as_tensor(y_train, dtype=DTYPE).to(DEVICE)
        X_test_tensor = torch.as_tensor(X_test, dtype=DTYPE).to(DEVICE)
        y_test_tensor = torch.as_tensor(y_test, dtype=DTYPE).to(DEVICE)
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=True
        )
        return train_loader, test_loader

    def train(self, round_num):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.005)
        self.net.train()
        if self.mode == "attack" and round_num == ROUND_MAX and type == "gan":
           self.fake_data, self.fake_labels = self.load_fake_data()
           self.batch_size = len(self.fake_data) // (TRAIN_EPOCHS * (ROUND_MAX - ROUND - 1))
           self.fake_data_batches = torch.split(self.fake_data, self.batch_size)
           self.fake_labels_batches = torch.split(self.fake_labels, self.batch_size) 
        for epoch in range(TRAIN_EPOCHS):
            # Attack mode  - GAN model poisoning - phase 2 - Model Training
            if self.mode == "attack" and round_num > ROUND and round_num >= ROUND_MAX and type == "gan":
                self.index = (self.index + 1) % (TRAIN_EPOCHS * (ROUND_MAX - ROUND - 1)) 
                batch_data = self.fake_data_batches[self.index]
                batch_labels = self.fake_labels_batches[self.index]
                poisoned_trainloader = DataLoader(
                    TensorDataset(batch_data, batch_labels), batch_size=32, shuffle=True
                )
                for inputs, labels in poisoned_trainloader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = criterion(outputs, torch.argmax(labels, dim=1))
                    loss.backward()
                    # for _, param in self.net.named_parameters():
                    #     param.grad.data *= 2  # Scale factor for fake data
                    optimizer.step()          
            else: 
                for inputs, labels in self.trainloader:
                # Attack mode  - GAN model poisoning - phase 1 - GAN Training
                    if self.mode == "attack" and round_num < ROUND_MAX and type == "gan":
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        # Training the model on real training set - reducing detection
                        optimizer.zero_grad()
                        outputs = self.net(inputs)
                        loss = criterion(outputs, torch.argmax(labels, dim=1))
                        loss.backward()
                        optimizer.step()
                        self.train_gan(inputs, labels)
                    else:
                        # Attack mode  - label flipping
                        if self.mode == "attack" and round_num > ROUND and type == "label_flip":
                            poisoned_labels = labels.clone()
                            poisoned_labels[:, [1, 2]] = poisoned_labels[:, [2, 1]] # flipping one-hot encoded labels
                            labels = poisoned_labels   
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        optimizer.zero_grad()
                        outputs = self.net(inputs)
                        loss = criterion(outputs, torch.argmax(labels, dim=1))
                        loss.backward()
                        # Attack mode  - gradient sign flipping
                        if self.mode == "attack" and round_num > ROUND and type == "gradient_flip":
                            for _, para in self.net.named_parameters():
                                para.grad.data = -para.grad.data
                        optimizer.step()
                    
    def train_gan(self, real_data, real_labels):
        batch_size = real_data.size(0)

        # self.load_gan_parameters()
        # Train discriminator on real data
        self.optimizer_d.zero_grad()
        real_output = self.discriminator(real_data)
        real_labels = torch.ones((batch_size, 1), dtype=DTYPE).to(DEVICE) # Class 0 for real data
        loss_real = self.criterion_d(real_output, real_labels)
        loss_real.backward()

        # Fake data generation
        self.optimizer_g.zero_grad()       
        z = torch.randn(batch_size, 100, dtype=DTYPE).to(DEVICE)
        # z[:, 1:] = z[:, 0:1]
        fake_data = self.generator(z)
        fake_labels = torch.zeros((batch_size, 1), dtype=DTYPE).to(DEVICE) # Class 1 for fake data

        # Train discriminator on fake data
        fake_output = self.discriminator(fake_data.detach())
        loss_fake = self.criterion_d(fake_output, fake_labels)
        loss_fake.backward()
        self.optimizer_d.step()

        # Train generator
        fake_output = self.discriminator(fake_data)
        loss_g = self.criterion_g(fake_output, real_labels)
        loss_g.backward()
        self.optimizer_g.step()

        # Save fake data
        fake_classes = torch.randint(0, 3, (batch_size,)).to(DEVICE)  # Randomly pick fake classes
        fake_labels = torch.zeros((batch_size, 3), dtype=torch.float64).to(DEVICE) 
        fake_labels.scatter_(1, fake_classes.view(-1, 1), 1.0)  # Convert to one-hot encoding
      
        self.save_fake_data(fake_data, fake_labels)
        # self.save_gan_parameters()

        # print(f"GAN Training for Client {self.client_id} - Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    def test(self, loader):
        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.net.eval()
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                target = torch.argmax(target, dim=1)
                outputs = self.net(data)
                loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        loss /= len(loader.dataset)
        accuracy = correct / total
        return loss, accuracy

    def save_fake_data(self, fake_data, fake_labels):
        fake_data_path = f"fake_data/client_{self.client_id}.pkl"
        os.makedirs("fake_data", exist_ok=True)
        if os.path.exists(fake_data_path):
            with open(fake_data_path, "rb") as f:
                existing_fake_data, existing_fake_labels = pickle.load(f)
            fake_data = torch.cat((existing_fake_data, fake_data), axis=0)
            fake_labels = torch.cat((existing_fake_labels, fake_labels), axis=0)
        else:
            fake_data = fake_data.detach().cpu()
            fake_labels = fake_labels.detach().cpu()

        # Add noise to the fake data for classes 1 and 2
        class_1_indices = (fake_labels[:, 1] == 1).nonzero(as_tuple=True)[0]
        class_2_indices = (fake_labels[:, 2] == 1).nonzero(as_tuple=True)[0]
        noise = torch.normal(0, 2, fake_data.shape[1:]).to(fake_data.device) # gaussian noise
        fake_data[class_1_indices] += noise
        fake_data[class_2_indices] -= noise

        with open(fake_data_path, "wb") as f:
            pickle.dump((torch.as_tensor(fake_data), torch.as_tensor(fake_labels)), f)
        print(f"Saved fake data for Client {self.client_id} at {fake_data_path}")

    def load_fake_data(self):
        fake_data_path = f"fake_data/client_{self.client_id}.pkl"
        with open(fake_data_path, "rb") as f:
            fake_data, fake_labels = pickle.load(f)
        return fake_data, fake_labels

def countdown(t):
    for i in range(t, 0, -1):
        print(i)
        sleep(1)


def client_wrapper(client_id, shared_dict, malicious):
    client = FlowerClient(client_id, mode, shared_dict, malicious).to_client()
    countdown(2)
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()
    clients = []
    for i in range(NUM_CLIENTS):
        p = Process(target=client_wrapper, args=(i, shared_dict, mode == "attack"))
        p.start()
        clients.append(p)
    for client in clients:
        client.join()
