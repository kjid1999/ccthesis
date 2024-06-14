import sys
sys.path.append('..')

def get_importance_estimate_model():
    '''
    return a BERT model to estimate tokens' importance
    '''
    bert_config = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_config)
    model = BertModel.from_pretrained(bert_config)
    model.eval()
    model.to('cuda')
    return model

if __name__ == '__main__':
    model = get_importance_estimate_model()