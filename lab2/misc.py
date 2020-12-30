import numpy as np
import nltk
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

__all__ = ['Metrics', 'evaluate']

PAD_ID, SOS_ID, EOS_ID, UNK_ID = [0, 1, 2, 3]


class Metrics:
    """
    """

    def __init__(self):
        super(Metrics, self).__init__()

    def sim_bleu(self, hyps, ref):
        """
        :param ref - a list of tokens of the reference
        :param hyps - a list of tokens of the hypothesis

        :return maxbleu - recall bleu
        :return avgbleu - precision bleu
        """
        scores = []
        for hyp in hyps:
            try:
                scores.append(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method7,
                                            weights=[1. / 4, 1. / 4, 1. / 4, 1. / 4]))
            except:
                scores.append(0.0)
        return np.max(scores), np.mean(scores)


def sent2indexes(sentence, vocab, maxlen):
    '''sentence: a string or list of string
       return: a numpy array of word indices
    '''

    def convert_sent(sent, vocab, maxlen):
        idxes = np.zeros(maxlen, dtype=np.int64)
        idxes.fill(PAD_ID)
        tokens = nltk.word_tokenize(sent.strip())
        idx_len = min(len(tokens), maxlen)
        for i in range(idx_len): idxes[i] = vocab.get(tokens[i], UNK_ID)
        return idxes, idx_len

    if type(sentence) is list:
        inds, lens = [], []
        for sent in sentence:
            idxes, idx_len = convert_sent(sent, vocab, maxlen)
            # idxes, idx_len = np.expand_dims(idxes, 0), np.array([idx_len])
            inds.append(idxes)
            lens.append(idx_len)
        return np.vstack(inds), np.vstack(lens)
    else:
        inds, lens = sent2indexes([sentence], vocab, maxlen)
        return inds[0], lens[0]


def indexes2sent(indexes, vocab, ignore_tok=PAD_ID):
    '''indexes: numpy array'''

    def revert_sent(indexes, ivocab, ignore_tok=PAD_ID):
        toks = []
        length = 0
        indexes = filter(lambda i: i != ignore_tok, indexes)
        for idx in indexes:
            toks.append(ivocab[idx])
            length += 1
            if idx == EOS_ID:
                break
        return ' '.join(toks), length

    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim == 1:  # one sentence
        return revert_sent(indexes, ivocab, ignore_tok)
    else:  # dim>1
        sentences = []  # a batch of sentences
        lens = []
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens


def evaluate(model, metrics, test_loader, vocab, repeat, f_eval):
    ivocab = {v: k for k, v in vocab.items()}
    #     device = next(model.parameters()).device
    device = torch.device('cuda:0')

    recall_bleus, prec_bleus, avg_lens = [], [], []

    dlg_id = 0
    for context, context_lens, utt_lens, floors, response, res_lens in tqdm(test_loader):

        if dlg_id > 5000: break

        #        max_ctx_len = max(context_lens)
        max_ctx_len = context.size(1)
        max_utt_len = context.size(2)
        # context, utt_lens, floors = context[:, :max_ctx_len, 1:], utt_lens[:, :max_ctx_len] - 1, floors[:, :max_ctx_len]
        # remove empty utts and the sos token in the context and reduce the context length
        ctx, ctx_lens = context, context_lens
        #         context, context_lens, utt_lens \
        #             = [tensor.to(device) for tensor in [context, context_lens, utt_lens]]
        context, context_lens, utt_lens, floors \
            = [tensor.to(device) for tensor in [context, context_lens, utt_lens, floors]]

        #################################################
        # utt_lens[utt_lens == 0] = 1
        #################################################

        with torch.no_grad():
            # sample_words, sample_lens = model.sample(context, context_lens, utt_lens, repeat)
            sample_words, sample_lens = model.sample(context, context_lens, utt_lens, floors, repeat, max_utt_len)
        sample_words = sample_words.cpu().numpy()
        sample_lens = sample_lens.cpu().numpy()
        # nparray: [repeat x seq_len]

        pred_sents, _ = indexes2sent(sample_words, vocab)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        ref_str, _ = indexes2sent(response[0].numpy(), vocab, SOS_ID)
        # ref_str = ref_str.encode('utf-8')
        ref_tokens = ref_str.split(' ')

        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)

        avg_lens.append(np.mean(sample_lens))

        response, res_lens = [tensor.to(device) for tensor in [response, res_lens]]

        ## Write concrete results to a text file
        dlg_id += 1
        if f_eval is not None:
            f_eval.write("Batch {:d} \n".format(dlg_id))
            # print the context
            start = np.maximum(0, ctx_lens[0] - 5)
            for t_id in range(start, ctx_lens[0], 1):
                context_str = indexes2sent(ctx[0, t_id].numpy(), vocab)
                f_eval.write("Context {:d}-{:d}: {}\n".format(t_id, floors[0, t_id], context_str))
            # print the ground truth response
            f_eval.write("Target >> {}\n".format(ref_str.replace(" ' ", "'")))
            for res_id, pred_sent in enumerate(pred_sents):
                f_eval.write("Sample {:d} >> {}\n".format(res_id, pred_sent.replace(" ' ", "'")))
            f_eval.write("\n")
    prec_bleu = float(np.mean(prec_bleus))
    recall_bleu = float(np.mean(recall_bleus))
    result = {'avg_len': float(np.mean(avg_lens)),
              'recall_bleu': recall_bleu, 'prec_bleu': prec_bleu,
              'f1_bleu': 2 * (prec_bleu * recall_bleu) / (prec_bleu + recall_bleu + 10e-12),
              }

    if f_eval is not None:
        for k, v in result.items():
            f_eval.write(str(k) + ':' + str(v) + ' ')
        f_eval.write('\n')
    print("Done testing")
    print(result)

    return result
