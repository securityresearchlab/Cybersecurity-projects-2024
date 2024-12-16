import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg
from flwr.client.mod import fixedclipping_mod
from flwr.simulation import run_simulation


# Federated Learning Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)


# GAN: Generator Network

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output




DTYPE = torch.float32

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
device = "cpu"
batch_size_gan = 32
SEED = 1


#########################################
# Simulation configuration

# Number of nodes that run a ClientApp (one will be always malicious)
client_nodes = 2

# Number of round of federated learning
number_server_round = 15

# Number of epoch for the training used by each client
n_epochs = 5

# Number of round of samples generation run by the attacker (every round generates 32 fake samples)
generated_round = 10000 # must be even!

private_class = 9

# Activate client-side differential privacy
differential_privacy = False

# Decide if you want to save some fake samples generated in the last round
save_fake_images = True

#########################################


torch.backends.cudnn.enabled = False
torch.manual_seed(SEED)

transform = torchvision.transforms.Compose([
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize((0.1307,), (0.3081,))
              ])

class Client(NumPyClient):

  def __init__(self, client_id):
    self.network = Net()
    torch.manual_seed(SEED)
    self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate, momentum=momentum)
    self.client_id = client_id
    self.round = 0

    self.train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transform), 
                                                    batch_size=batch_size_train, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./mnist', train=False, download=True, transform=transform), 
                                                    batch_size=batch_size_test, shuffle=True)

    if self.client_id == 0:
      self.generator = Generator().to(device=device)
      self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
      self.loss_function = nn.BCELoss()
    

  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in self.network.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(self.network.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.as_tensor(v, dtype=DTYPE) for k, v in params_dict})
    self.network.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.round = self.round + 1
    if self.client_id == 0: # attacker
      self.set_parameters(parameters)
      self.attack()
    else: # victim
      self.set_parameters(parameters)
      round_number = config.get("server_round", 0)
      self.train(round_number)
    # self.save_model_weights(round_number)
    return self.get_parameters(config={}), len(self.train_loader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = self.test()
    print(f"[CLIENT {self.client_id}] Test Loss: {loss}, Test Accuracy: {accuracy}")
    return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}

  def train(self, round_number):
    self.network.train()
    for epoch in range(n_epochs):
      print('\033[32m[CLIENT {}] Train Epoch: {} [on {} samples]\033[m'.format(self.client_id, epoch, len(self.train_loader.dataset)))
      for batch_idx, (data, target) in enumerate(self.train_loader):
        self.optimizer.zero_grad()
        output = self.network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
      print("\033[32m\tLoss: {:.6f}\033[m".format(loss.item()))

  def attack(self):
    expected_samples_labels = torch.ones((batch_size_gan, 1)).to(device=device)
    fake_samples = []

    for epoch in range(generated_round):
      # Training the generator
      self.generator.zero_grad()
      # Generate fake samples with the generator
      latent_space_samples = torch.randn((batch_size_gan, 100)).to(device=device)
      generated_samples = self.generator(latent_space_samples)
      fake_samples.append(generated_samples)
      
      # generated_samples should be classified of class "private_class"
      output = self.network(generated_samples)
      probabilities = torch.softmax(output, dim=1)
      # Extract only the probabilites for class "private_class"
      generated_samples_pred = probabilities[:, private_class]  

      pred_tensor = generated_samples_pred.type(dtype=torch.float32).to(device).requires_grad_(True).unsqueeze(1)
      loss_generator = self.loss_function(
          pred_tensor, expected_samples_labels
      )
      loss_generator.backward()
      self.optimizer_generator.step()

      # save/show the generated fake image
      if save_fake_images and epoch == generated_round-1:
        os.makedirs("./fake_samples", exist_ok=True)
        image = generated_samples[0, 0, :, :]  
        plt.imsave(os.path.join("./fake_samples", f"fake_{private_class}_sample.png"), image.detach().cpu().numpy(), cmap="gray")
        plt.imshow(image.detach().cpu().numpy(), cmap='gray')
        plt.title("Generated Image")
        plt.show()
      elif not save_fake_images and (epoch == generated_round-1 or epoch == generated_round/2):
        image = generated_samples[0, 0, :, :]  # Extract the first image of the batch
        image = image.reshape(28, 28)
        plt.imshow(image.detach().cpu().numpy(), cmap='gray')
        plt.title("Generated Image")
        plt.show()

      if epoch == generated_round-1:
        print("\033[31m[CLIENT {}] GAN accuracy = {}\033[0m".format(self.client_id, torch.mean(generated_samples_pred)))

    # Train the model with the fake generated samples, but labeling it with a wrong class: shared_class
    if self.round > 1:
      self.network.train()
      
      target = torch.randint(0, 9, (64, 1))         
      target[target >= private_class] += 1

      for i in range(0, len(fake_samples), 2):
        if i + 1 < len(fake_samples):  
          train_samples = torch.cat((fake_samples[i], fake_samples[i + 1]), dim=0)
          self.optimizer.zero_grad()
          output = self.network(train_samples)
          loss = F.nll_loss(output, target)
          loss.backward()
          self.optimizer.step()
        print("\033[31m    Loss: {:.6f}\033[0m".format(loss.item()))
      

  def test(self):
    self.network.eval
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in self.test_loader:
        output = self.network(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(self.test_loader.dataset),
      100. * correct / len(self.test_loader.dataset)))
    accuracy = correct / len(self.test_loader.dataset)
    return test_loss, accuracy


def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return Client(partition_id).to_client()


# Create strategy
if differential_privacy:
    strategy = DifferentialPrivacyClientSideFixedClipping(FedAvg(), 0.01, 0.1 , client_nodes)
else:
    strategy = FedAvg()

def server_fn(context: Context) -> ServerAppComponents:
    config = ServerConfig(num_rounds=number_server_round)
    return ServerAppComponents(strategy=strategy, config=config)


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Create the ClientApp
if not differential_privacy:
    client = ClientApp(client_fn=client_fn)
else:
    client = ClientApp(client_fn=client_fn, mods=[fixedclipping_mod])

    
# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=client_nodes,
    backend_config=backend_config,
)
