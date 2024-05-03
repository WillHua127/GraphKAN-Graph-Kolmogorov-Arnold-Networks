# GraphKAN

Implementation of Graph Neural Network version of Kolmogorov Arnold Networks (GraphKAN)

The only dependencies are torch and pytorch_geometric.
If the gpu (cuda) running fails, change to cpu training by changing:
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ------> args.device = torch.device('cpu')

Thanks to the original implementations KAN(https://github.com/KindXiaoming/pykan) and FourierKAN (https://github.com/GistNoesis/FourierKAN), you guys are amazing.

Still at experimental stage to see if KAN really works on graph-structured data.

Two key experimental observations:
(1) It is way much better to implment KAN for graphs in the latent feature space.
That is having a Linear Layer to to project the input feature first: self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
Rather than directly using the KAN Layer for input features like: self.lin_in = KANLayer(in_feat, hidden_feat, grid_feat, addbias=use_bias)
Dont know if the trick works for CV. NLP. etc.

(2) Using SGD (ASGD) optimizer is way more stable than using Adam (AdamW) optimizer, but also SDG is really slow to converage, 
e.g., on Cora Adam takes ~200 epochs, SGD takes ~10000 epochs.
