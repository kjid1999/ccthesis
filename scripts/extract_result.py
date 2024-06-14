import re
from collections import namedtuple
import matplotlib.pyplot as plt

Block = namedtuple('Block', 'bleu_score, rouge_score, bert_score, dist_score, avg_len, ckpt_n') 

def process_mbr_file(path):
    # 讀取文本檔案
    assert 'mbr' in path, 'MBR result'
    print("read file", path)
    with open(path, "r") as file:
        data = file.read()

    blocks = data.split('\n\n')

    # 定義正則表達式來匹配區塊
    pattern = """
.*?avg BLEU score (\d+\.\d+)
.*?avg ROUGE-L score (\d+\.\d+)
.*?avg berscore tensor\((\d+\.\d*)\)
.*?avg dist1 score (\d+\.\d+)
.*?avg len (\d+\.\d+)
.*?(\d+).pt.*?
"""

    matches = [re.findall(pattern, bolck, re.DOTALL) for bolck in blocks]
    matches = [[float(m_in) for m_in in m[0]] for m in matches if m != []]

    blocks = [Block(*m) for m in matches]

    # print(data)
    # print(matches)
    # print(blocks[0])

    return blocks

def process_non_mbr_file(path):
    # 讀取文本檔案
    assert 'mbr' not in path, 'MBR result'
    print("read file", path)
    with open(path, "r") as file:
        data = file.read()

    blocks = data.split('\n\n')

    # 定義正則表達式來匹配區塊
    pattern = """.*?avg BLEU score (\d+\.\d+)
.*?avg ROUGE-L score (\d+\.\d+)
.*?avg berscore tensor\((\d+\.\d*)\)
.*?avg dist1 score (\d+\.\d+)
.*?avg len (\d+\.\d+)
"""

    patten2 = '.*ema_0\.9999_(\d+)\.pt\.samples'

    matches = [re.findall(pattern, bolck) for bolck in blocks]
    matches = [[float(m_in) for m_in in m[0]] for m in matches if m]
    matches2 = [re.findall(patten2, bolck) for bolck in blocks]
    matches2 = [int(m[0]) for m in matches2 if m]

    blocks = [Block(*m, n) for m, n in zip(matches, matches2)]

    # print(data)
    # print(matches)
    # print(matches2)
    # print(blocks[-1].ckpt_n)

    return blocks

def auto_process_file(path):
    if 'mbr' in path:
        return process_mbr_file(path)
    else:
        return process_non_mbr_file(path)

def draw_save(xs, ys, fig_name):
    ax = plt.subplot()
    for x, y in zip(xs, ys):
        ax.plot(x, y)
    plt.savefig(fig_name)

def find_best(blocks):
    best_blue = max(blocks, key=lambda b: float(b.bleu_score))
    best_rouge = max(blocks, key=lambda b: float(b.rouge_score))
    best_bert = max(blocks, key=lambda b: float(b.bert_score))

    print(f"{'':20s}|{'best BLUE':<20s} | {'best Rouge':<20} | {'best bertscore':<20s}")
    print('='*100)
    print(f"{'checkpoint':20s}|{best_blue.ckpt_n:20d} | {best_rouge.ckpt_n:20d} | {best_bert.ckpt_n:20d}")
    print(f"{'bleu_score':20s}|{best_blue.bleu_score:20.4f} | {best_rouge.bleu_score:20.4f} | {best_bert.bleu_score:20.4f}")
    print(f"{'rouge_score':20s}|{best_blue.rouge_score:20.4f} | {best_rouge.rouge_score:20.4f} | {best_bert.rouge_score:20.4f}")
    print(f"{'bert_score':20s}|{best_blue.bert_score:20.4f} | {best_rouge.bert_score:20.4f} | {best_bert.bert_score:20.4f}")


if __name__ == '__main__':
    eval_result_mbr = auto_process_file('eval_result_mbr.txt')
    eval_result_step2 = auto_process_file('eval_result_step2.txt')
    eval_result_step10 = auto_process_file('eval_result_step10.txt')
    eval_result_step20 = auto_process_file('eval_result_step20.txt')
    eval_result_step_20_pretrain = auto_process_file('eval_result_step_20_pretrain.txt')
    eval_result_step_20_lambda01= auto_process_file('eval_result_step20_lambda0.1.txt')
    eval_result_step20_lambda0= auto_process_file('eval_result_step20_lambda0.txt')
    eval_result_step20_lambda0_black= auto_process_file('eval_result_step20_lambda0_???.txt')
    eval_result_step20_lambda01_correct_mask= auto_process_file('eval_result_step20_lambda0.1_correct_mask.txt')
    eval_result_step20_lambda05_correct_mask = auto_process_file('eval_result_step20_lambda0.5_correct_mask.txt')
    # eval_result_step20_lambda05_importance_mean = auto_process_file('eval_result_step20_lambda0.5_importance-mean.txt')
    eval_result_step0_lambda0_importance_mean = auto_process_file('eval_result_step0_lambda0_importance-mean.txt')
    eval_result_step0_lambda05_correct_mask = auto_process_file('eval_result_step0_lambda0.5_correct_importance_mask.txt')
    eval_result_step20_lambda0_maxlen98 = auto_process_file('eval_result_step20_lambda0_maxlen98.txt')

    # find_best(eval_result_step20)
    # find_best(eval_result_step_20_lambda01)
    # find_best(eval_result_step20_lambda0)
    # find_best(eval_result_step20_lambda01_correct_mask)
    # find_best(eval_result_step20_lambda05_correct_mask)
    # find_best(eval_result_step20_lambda05_correct_mask)
    # find_best(eval_result_step20_lambda05_importance_mean) # best in lambda05
    # find_best(eval_result_step0_lambda0_importance_mean)
    # find_best(eval_result_step0_lambda05_correct_mask)
    find_best(eval_result_step20_lambda0_maxlen98)

    exit()

    all_files = [
        eval_result_mbr,
        eval_result_step2,
        eval_result_step10,
        eval_result_step20,
        eval_result_step_20_pretrain
    ]

    bleu_figure, bleu_ax = plt.subplots()
    rouge_figure, rouge_ax = plt.subplots()
    bert_figure, bert_ax = plt.subplots()
    for file in all_files:
        x = [f.ckpt_n for f in file]
        blue = [float(f.bleu_score) for f in file]
        rouge = [float(f.rouge_score) for f in file]
        bert = [float(f.bert_score) for f in file]

        bleu_ax.plot(x, blue)
        rouge_ax.plot(x, rouge)
        bert_ax.plot(x, bert)

    bleu_figure.savefig('bleu_figure.png')
    rouge_figure.savefig('rouge_figure.png')
    bert_figure.savefig('bert_figure.png')
