import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .transformer import TransformerDecoder, TransformerDecoderLayer
from .blocks import ModifiedResNet, PMC_CLIP_cfg

class QA_model_infer(nn.Module):
    """
    A simplified QA_model for 4-bit quantized inference.
    It accepts a pre-loaded PEFT model and handles the vision components.
    """
    def __init__(self, llm, pmc_clip_ckpt_path, img_tokens=32, H=8, N=12):
        super().__init__()
        self.llamacasual = llm
        self.device = llm.device
        self.dtype = torch.float16
        self.img_tokens = img_tokens

        # --- Vision Model ---
        vision_cfg = PMC_CLIP_cfg()
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        vision_model = ModifiedResNet(
            layers=vision_cfg.layers, heads=vision_heads, output_dim=768,
            image_size=vision_cfg.image_size, width=vision_cfg.width
        )
        checkpoint = torch.load(pmc_clip_ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.visual.', ''): v for k, v in state_dict.items() if '.visual' in k}
        vision_model.load_state_dict(state_dict)
        self.vision_model = nn.Sequential(*list(vision_model.children())[:-2])
        self.vision_model.to(device=self.device, dtype=self.dtype)
        num_ftrs = 1024  # ResNet feature dimension

        # --- Q-Former / Projection Module ---
        self.query_embed = nn.Embedding(img_tokens, num_ftrs)
        decoder_layer = TransformerDecoderLayer(num_ftrs, H, 1024, 0.1, 'relu', normalize_before=True)
        decoder_norm = nn.LayerNorm(num_ftrs)
        self.decoder = TransformerDecoder(decoder_layer, N, decoder_norm, return_intermediate=False)
        self.fc_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.fc_l2 = nn.Linear(num_ftrs, llm.config.hidden_size)
        
        # Move all non-LLM parts to the correct device and dtype
        self.query_embed.to(device=self.device, dtype=self.dtype)
        self.decoder.to(device=self.device, dtype=self.dtype)
        self.fc_l1.to(device=self.device, dtype=self.dtype)
        self.fc_l2.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, input_ids, images, attention_mask):
        B = images.shape[0]
        # Image Encoding and Projection
        img_feat_map = self.vision_model(images)
        features = rearrange(img_feat_map, 'b d n1 n2 -> b (n1 n2) d').transpose(0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        features, _ = self.decoder(query_embed, features)
        features = rearrange(features.transpose(0, 1), 'b n d -> (b n) d')
        features = self.fc_l2(F.relu(self.fc_l1(features)))
        features = rearrange(features, '(b n) d -> b n d', b=B)
        
        # Prepare embeddings for the LLM
        txt_embed = self.llamacasual.get_input_embeddings()(input_ids)
        input_embedding = torch.cat([features.to(txt_embed.dtype), txt_embed], dim=1)
        
        # Create a combined attention mask for image and text
        image_atts = torch.ones(features.size()[:-1], dtype=torch.long).to(self.device)
        full_attention_mask = torch.cat([image_atts, attention_mask], dim=1)
        
        # Generate text using autocast for mixed-precision stability
        with torch.cuda.amp.autocast(dtype=torch.float16):
            generation = self.llamacasual.generate(
                inputs_embeds=input_embedding,
                attention_mask=full_attention_mask,
                max_new_tokens=50
            )
        return generation