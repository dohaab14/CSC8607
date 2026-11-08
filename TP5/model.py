import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


resnet_pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# On garde la carte spatiale (pas de vecteur global) : on enlève AdaptiveAvgPool2d et Linear
cnn_features = nn.Sequential(*list(resnet_pretrained.children())[:-2])

# Transfer learning : on gèle les poids
for param in cnn_features.parameters():
    param.requires_grad = False


class AttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(AttentionModule, self).__init__()
        # Couche pour calculer le score d'attention (un score par zone)
        self.attention_layer = nn.Conv2d(feature_dim + hidden_dim, 1, 1)

    def forward(self, features, hidden_state):
        # features : (batch_size, 2048, 7, 7) ; hidden_state : (batch_size, hidden_dim)

        # On "étale" l'état caché sur la grille spatiale 7x7
        hidden_state_expanded = hidden_state.unsqueeze(2).unsqueeze(3).repeat(
            1, 1, features.size(2), features.size(3))

        # Concaténation sur la dimension des canaux (dim=1)
        concat_features = torch.cat((features, hidden_state_expanded), dim=1)

        # Scores (batch_size, 1, 7, 7) -> on retire la dimension de taille 1
        attention_scores = self.attention_layer(concat_features).squeeze(1)

        # Softmax sur les 49 régions (7x7)
        attention_weights = F.softmax(attention_scores.view(attention_scores.size(0), -1), dim=1)
        attention_weights = attention_weights.view(
            attention_scores.size(0), 1, attention_scores.size(1), attention_scores.size(2))

        # Somme pondérée sur H et W (dim 3 puis dim 2)
        context_vector = torch.sum(torch.sum(features * attention_weights, dim=3), dim=2)

        return context_vector, attention_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_module, attention_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_module = attention_module

        # Entrée des portes : mot + état caché précédent + contexte d'attention
        total_dim = input_size + hidden_size + attention_size

        self.lin_i = nn.Linear(total_dim, hidden_size)  # porte d'entrée
        self.lin_f = nn.Linear(total_dim, hidden_size)  # porte d'oubli
        self.lin_c = nn.Linear(total_dim, hidden_size)  # candidat mémoire
        self.lin_o = nn.Linear(total_dim, hidden_size)  # porte de sortie
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, features, init_states=None):
        # x : (batch_size, sequence_size, input_size)
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(bs, self.hidden_size).to(x.device)
            c_t = torch.zeros(bs, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            # Contexte d'attention calculé avec l'état caché courant
            att_t, _ = self.attention_module(features, h_t)

            # Concaténation : contexte, mot actuel, état caché précédent
            concat = torch.cat((att_t, x_t, h_t), dim=1)

            # Équations du LSTM
            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            # Mise à jour de la mémoire et de l'état caché
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)  # (batch_size, seq_size, hidden_size)
        return hidden_seq, (h_t, c_t)



class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim):
        """cnn_model = le ResNet50 COMPLET (le [:-2] est appliqué ici, une seule fois)."""
        super().__init__()
        self.cnn = nn.Sequential(*list(cnn_model.children())[:-2])
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.attention = AttentionModule(feature_dim=2048, hidden_dim=lstm_hidden_dim)
        self.embeddings = embedding_model
        vocab_size = embedding_model.weight.shape[0]

        # Entrée du LSTM = dimension de l'embedding word2vec (300)
        emb_dim = embedding_model.weight.shape[1]
        self.lstm = LSTMWithAttention(emb_dim, lstm_hidden_dim, self.attention, 2048)

        # Projection finale : sortie du LSTM -> un score par mot du vocabulaire
        self.fc_out = nn.Linear(lstm_hidden_dim, vocab_size)

    def train(self, mode=True):
        super().train(mode)
        self.cnn.eval()  # ResNet gelé : BatchNorm reste en mode inférence
        return self

    def forward(self, images, captions):
        with torch.no_grad():                   # pas de gradient dans le ResNet gelé
            features = self.cnn(images)         # (B, 2048, 7, 7)
        emb = self.embeddings(captions)         # (B, T, 300)
        lstm_outputs, (hidden_state, _) = self.lstm(emb, features)
        return self.fc_out(lstm_outputs)        # (B, T, vocab_size)



if __name__ == "__main__":
    # Exercice 2
    x = torch.randn(32, 3, 224, 224)
    print("ResNet sortie :", cnn_features(x).shape)
    n_train = sum(p.numel() for p in cnn_features.parameters() if p.requires_grad)
    print("paramètres entraînables du ResNet :", n_train)

    # Attention
    att = AttentionModule(feature_dim=2048, hidden_dim=128)
    features = torch.randn(4, 2048, 7, 7)
    h = torch.randn(4, 128)
    context, weights = att(features, h)
    print("context :", context.shape)
    print("poids   :", weights.shape)
    print("somme des poids (image 0) :", weights[0].sum().item())

    # LSTM
    lstm = LSTMWithAttention(input_size=300, hidden_size=128, attention_module=att, attention_size=2048)
    xs = torch.randn(4, 19, 300)
    hidden_seq, (h_t, c_t) = lstm(xs, features)
    print("hidden_seq :", hidden_seq.shape)
    print("h_t, c_t   :", h_t.shape, c_t.shape)

    # Modèle global (ResNet non pré-entraîné ici, juste pour tester les formes)
    emb = nn.Embedding(50, 300)
    model = ImageCaptioningModel(models.resnet50(weights=None), emb, 128)
    out = model(torch.randn(2, 3, 224, 224), torch.randint(1, 50, (2, 19)))
    print("sortie modèle :", out.shape)