import torch
from transformers import BertTokenizer, BertModel
from diffuseq.utils import dist_util

def _get_attention_map(attention_weights):
    '''
    Convert tuple of layer attentions to a tensor
    attention_weights: a tuple with shape (num_layers,), 
        each element in this tuple is of shape (bsz, num_heads, tokens_len, tokens_len)
    return a tensor with shape (num_layers, bsz, num_heads, tokens_len, tokens_len)
    '''
    attention_map = torch.stack(attention_weights)
    return attention_map

@torch.no_grad()
def importance(input_ids, atten_mask, model, normalize=True):
    '''
    input_ids: input sentences token index, (B, T, D)
    model: BERT to calculate attention score
    '''
    model.config.output_attentions = True
    outputs = model(input_ids, atten_mask)
    attention_weights = outputs.attentions
    # print(input_ids.shape)
    # print(atten_mask.shape)

    attention_map = _get_attention_map(attention_weights)

    importance_score = attention_map[:, :, :, :, :].mean(dim=(0, 2, -2))

    if normalize:
        return importance_score - importance_score.mean(dim=-1, keepdim=True)
    else:
        return importance_score

def get_importance_estimate_model():
    '''
    return a BERT model to estimate tokens' importance
    '''
    bert_config = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_config)
    model = BertModel.from_pretrained(bert_config)
    model.eval()
    model.to(dist_util.dev())
    return model, tokenizer

def _S(t, T, _lambda=0.5):
    return _lambda*torch.sin(t*torch.pi / T)

def calculate_mask_rate(importance_score, t, num_timesteps, _lambda=0.5):
    T = num_timesteps
    t = t[..., None]
    return 1 - torch.clip(1 - t/T - _S(t, T, _lambda)*importance_score, 0, 1)

def get_word_freq():
    return torch.load('./word_freq/bert-base-uncased_qqp_nocount_special.pt')

# word_appear = torch.load('./word_freq/bert-base-uncased_qqp.pt')
word_appear = get_word_freq()
sum_appear = sum(word_appear)
@torch.no_grad()
def H(input_ids, target_mask):
    sen_len = target_mask.sum(dim=-1, keepdim=True)
    
    entropy = - torch.log(word_appear[input_ids].cuda() / sum_appear)
    entropy *= target_mask
    entropy = torch.nan_to_num(entropy, nan=0.)

    ret = 1 - entropy.sum(dim=-1, keepdim=True) / (sen_len * entropy)
    ret = torch.nan_to_num(ret, nan=0.)
    return target_mask * ret

def freq_rank():
    # print(type(word_appear)) # <class 'torch.Tensor'>  
    # print((word_appear).shape) # torch.Size([30522])

    ret = word_appear.argsort(descending=True)
    return ret

if __name__ == '__main__':
    freq_rank()