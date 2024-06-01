import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LowCostMultiModalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_modalities, dropout=0.1):
        super(LowCostMultiModalTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_modalities = num_modalities

        # Define the encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # Define the embedding for each modality
        self.embeddings = nn.ModuleList([
            nn.Linear(d_model, d_model) 
            for _ in range(num_modalities)
        ])
        
        self.fc_out = nn.Linear(d_model, num_modalities)

    def forward(self, *modal_inputs):
        # Embed each modality input
        embeddings = [self.embeddings[i](modal_inputs[i]) for i in range(self.num_modalities)]
        
        # Concatenate along the sequence dimension
        x = torch.cat(embeddings, dim=1)  # Assume inputs are of shape (batch, seq_len, d_model)
        
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        output = self.fc_out(x.mean(dim=1))  # Assuming classification task, output shape (batch, num_modalities)
        
        return output

class LowCostAttention(nn.Module):
    def __init__(self, d_model, nhead, num_modalities, dropout=0.1):
        super(LowCostAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_modalities = num_modalities

        self.self_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            for _ in range(num_modalities)
        ])

        self.cross_attention_heads = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            for _ in range(num_modalities * (num_modalities - 1))
        ])

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, *modal_inputs):
        batch_size = modal_inputs[0].size(0)
        
        # Self-attention for each modality
        self_attention_outputs = []
        for i in range(self.num_modalities):
            modal_input = modal_inputs[i]
            attn_output, _ = self.self_attention_heads[i](modal_input, modal_input, modal_input)
            self_attention_outputs.append(attn_output)
        
        # Cross-attention between modalities
        cross_attention_outputs = []
        for i in range(self.num_modalities):
            for j in range(i + 1, self.num_modalities):
                cross_modal_input1 = self_attention_outputs[i]
                cross_modal_input2 = self_attention_outputs[j]
                attn_output, _ = self.cross_attention_heads[i * (self.num_modalities - 1) + j - 1](
                    cross_modal_input1, cross_modal_input2, cross_modal_input2
                )
                cross_attention_outputs.append(attn_output)

        # Combine self-attention and cross-attention outputs
        combined_output = torch.cat(self_attention_outputs + cross_attention_outputs, dim=1)
        
        # Final output layer
        output = self.fc_out(combined_output.mean(dim=1))  # Assuming classification task, output shape (batch, d_model)
        
        return output


class LoCoMTGenerator(nn.Module):
    def __init__(self, latent_dim, d_model, nhead, num_layers, num_modalities, img_size, num_channels=3, dropout=0.1):
        super(LoCoMTGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_channels = num_channels

        self.locomt = LowCostMultiModalTransformer(d_model, nhead, num_layers, num_modalities, dropout)

        self.fc = nn.Linear(d_model, img_size * img_size * num_channels)
        self.tanh = nn.Tanh()

    def forward(self, noise, *modal_inputs):
        locomt_output = self.locomt(*modal_inputs)
        combined_input = torch.cat((locomt_output, noise), dim=1)
        img = self.fc(combined_input)
        img = img.view(img.size(0), self.num_channels, self.img_size, self.img_size)
        return self.tanh(img)

class LoCoMTDiscriminator(nn.Module):
    def __init__(self, img_size, num_channels=3, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(LoCoMTDiscriminator, self).__init__()
        self.img_size = img_size
        self.num_channels = num_channels

        self.conv = nn.Conv2d(num_channels, d_model, kernel_size=4, stride=2, padding=1)
        self.locomt = LowCostMultiModalTransformer(d_model, nhead, num_layers, 1, dropout)  # Single modality (image)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.conv(img).view(img.size(0), -1, self.img_size // 2)  # Adjust dimensions as needed
        locomt_output = self.locomt(x)
        validity = self.fc(locomt_output.mean(dim=1))
        return self.sigmoid(validity)


import torch.optim as optim


# Hyperparameters
latent_dim = 100
d_model = 512
nhead = 8
num_layers = 6
num_modalities = 2  # e.g., image and text
img_size = 64
num_channels = 3
dropout = 0.1

# Create generator and discriminator
generator = LoCoMTGenerator(latent_dim, d_model, nhead, num_layers, num_modalities, img_size, num_channels, dropout)
discriminator = LoCoMTDiscriminator(img_size, num_channels, d_model, nhead, num_layers, dropout)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    for i, (imgs, captions) in enumerate(dataloader):  # Assuming dataloader yields (image, text) pairs

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Configure input
        real_imgs = imgs.to(device)
        text_embeddings = text_encoder(captions).to(device)  # Assuming text_encoder gives text embeddings
        noise = torch.randn(batch_size, latent_dim).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(noise, text_embeddings)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real images
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        
        # Loss for fake images
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

print("Training finished.")