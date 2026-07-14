from torch import nn
import torch
from transformers import AutoTokenizer,AutoModel

class TextMotionCrossAttention(nn.Module):
  def __init__(self, encoder, dim=64, heads=1):
    #dim should be the size of the FaceFormer latent, default is 64
    super(TextMotionCrossAttention, self).__init__()
    self.encoder = encoder
    self.projector  = nn.Linear(768,dim)
    self.cross_attn = nn.MultiheadAttention(embed_dim=dim, batch_first=True,num_heads=heads)
  def forward(self, tokens, motion):
    input_ids = tokens["input_ids"].to(motion.device)
    attention_mask =  tokens["attention_mask"].to(motion.device)
    key_padding_mask = ~attention_mask.bool()
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    text_latent = self.projector(hidden_states)
    query = motion
    key   = text_latent
    value = text_latent
    attn_output, attn_output_weights = self.cross_attn(query, key, value,key_padding_mask=key_padding_mask)
    return motion + attn_output

if __name__=="__main__":
  device = "cpu"
  model_name = "bert-base-uncased"
  text = "The man looks happy"
  tokenizer  = AutoTokenizer.from_pretrained(model_name)
  tokens = tokenizer(text,return_tensors="pt")
  encoder = AutoModel.from_pretrained(model_name) 
  context_transformer = TextMotionCrossAttention(encoder=encoder, dim=64)
  contextualized_vertices = context_transformer(tokens, vertice_out)

#q = tgt  
#k = v = mem
#attn = k@q[:len(k)].T
#out = attn@v   