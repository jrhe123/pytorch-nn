# 1. Import Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Check for mps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Prepare the Data
text = "hello pytorch rnn example"

# Create a character dictionary
chars = list(set(text))
char2idx = {char: idx for idx, char in enumerate(chars)}
idx2char = {idx: char for idx, char in enumerate(chars)}

# Hyperparameters
input_size = len(chars)
hidden_size = 128
num_layers = 1
output_size = len(chars)
seq_length = 5
learning_rate = 0.01


# Prepare input and target sequences
def create_sequences(text, seq_length):
    sequences = []
    for i in range(len(text) - seq_length):
        seq_in = text[i : i + seq_length]
        seq_out = text[i + seq_length]
        sequences.append((seq_in, seq_out))
    return sequences


sequences = create_sequences(text, seq_length)


# Convert sequences to tensor format
def seq_to_tensor(seq, char2idx):
    tensor = torch.zeros(len(seq), dtype=torch.long)
    for i, char in enumerate(seq):
        tensor[i] = char2idx[char]
    return tensor


input_sequences = [seq_to_tensor(seq_in, char2idx) for seq_in, _ in sequences]
target_sequences = [char2idx[seq_out] for _, seq_out in sequences]

input_sequences = torch.stack(input_sequences).to(device)
target_sequences = torch.tensor(target_sequences).to(device)


# 3. Define the RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


model = RNN(
    input_size,
    hidden_size,
    output_size,
    num_layers,
).to(device)

# 4. Train the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 500
for epoch in range(num_epochs):
    hidden = model.init_hidden(input_sequences.size(0))
    outputs, hidden = model(input_sequences, hidden)
    loss = criterion(outputs, target_sequences)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# 5. Generate Text
def generate_text(model, start_str, length):
    model.eval()
    input_seq = seq_to_tensor(start_str, char2idx).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    generated_str = start_str

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        _, top_idx = torch.topk(output, 1)
        predicted_char = idx2char[top_idx.item()]
        generated_str += predicted_char
        input_seq = (
            seq_to_tensor(generated_str[-seq_length:], char2idx).unsqueeze(0).to(device)
        )

    return generated_str


# Generate new text
start_str = "example"
generated_text = generate_text(model, start_str, 20)
print(f"Generated Text: {generated_text}")
