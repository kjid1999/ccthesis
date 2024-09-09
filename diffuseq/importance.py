import torch
from transformers import BertTokenizer, BertModel
from diffuseq.utils import dist_util
from collections import Counter

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
def importance(input_ids, target_mask):
    '''
    input_ids: input sentences token index, (B, T)
    '''
    tf_idf_w = tf_idf(input_ids, target_mask) 
    H_w = H(input_ids, target_mask)
    ret = tf_idf_w/tf_idf_w.sum(dim=-1)[..., None] + H_w/H_w.sum(dim=-1)[..., None]
    return torch.nan_to_num(ret, nan=0.)

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
    return torch.load('./word_freq/bert-base-uncased_qqp_nocount_special.pt', map_location='cuda')

# word_appear = torch.load('./word_freq/bert-base-uncased_qqp.pt')
word_appear = get_word_freq()
sum_appear = sum(word_appear)
@torch.no_grad()
def H(input_ids, target_mask):
    sen_len = target_mask.sum(dim=-1, keepdim=True)
    
    p = word_appear[input_ids].cuda() / sum_appear
    entropy = - p * torch.log(p)
    entropy *= target_mask
    entropy = torch.nan_to_num(entropy, nan=0.)

    # ret = entropy.sum(dim=-1, keepdim=True) / (sen_len * entropy)
    # ret = torch.nan_to_num(ret, nan=0.)
    return target_mask * entropy


num_sen_word_appear = torch.load('./word_freq/bert-base-uncased_qqp_tf.pt', map_location='cuda')
@torch.no_grad()
def tf_idf(input_ids, target_mask):
    '''
    input_ids: B x L
    '''
    # sen_len = target_mask.sum(dim=-1, keepdim=True)
    l = []

    # exit()
    for d in input_ids.cpu():
        # c = Counter(d)
        keys, values = torch.unique(d, sorted=False, return_counts=True)
        h = dict([(k.item(), v) for k, v in zip(keys, values)])
        l.append(d.apply_(h.get))
    f = torch.stack(l).cuda() * target_mask
    N = 144715
    ret = f / f.sum(dim=-1)[:, None] * torch.log(N / (1 + num_sen_word_appear[input_ids]))
    ret = torch.nan_to_num(ret, nan=0.)
    return target_mask * ret

def freq_rank():
    # print(type(word_appear)) # <class 'torch.Tensor'>  
    # print((word_appear).shape) # torch.Size([30522])

    ret = word_appear.argsort(descending=True)
    return ret

if __name__ == '__main__':
    freq_rank()