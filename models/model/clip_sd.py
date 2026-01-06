import clip
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, c_in, c_out, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_out, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ClipCustom(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, cfg,num):
        super().__init__()
        self.device = "cuda"

        self.model, _ = clip.load("ViT-L/14", jit=False, device="cpu")
        use_adapter = True
        if use_adapter:
            self.adapter = Adapter(768, num, 1) # (768, 512, 1) # (768, 384, 1)
            self.use_adapter = True
        else:
            self.use_adapter = False

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text, pre_data=None):

        tokens = clip.tokenize(text).to(self.device)

        x = self.model.token_embedding(tokens).type(self.model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        if self.use_adapter:
            x = self.adapter(x)
        return x, tokens


if __name__ == "__main__":
    model = ClipCustom().to("cuda")
    cls_token, txt_token = model(["abcd","acv ooo"])