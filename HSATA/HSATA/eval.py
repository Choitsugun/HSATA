from __future__ import print_function
import codecs
import os
import argparse
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_en_vocab, load_tw_vocab
from train import Graph

def eval():
    # Load vocabulary
    token2idx, idx2token = load_de_en_vocab()
    tw2idx, idx2tw = load_tw_vocab()
    token2idx_len = len(token2idx)
    tw2idx_len = len(tw2idx)

    # Load vocab_overlap
    token_idx_list = []
    con_list = np.zeros([4, token2idx_len],dtype='float32')
    for i in range(4, tw2idx_len):
        tw = idx2tw[i]
        token_idx_list.append(token2idx[tw])

    vocab_overlap = np.append(con_list, np.eye(token2idx_len, dtype='float32')[token_idx_list], axis=0)

    # Load graph
    g = Graph(False, token2idx_len, tw2idx_len, vocab_overlap)
    print("Graph loaded")
    
    # Load data
    X, X_length, Y, TW, Sources, Targets = load_test_data()
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                args = parse_args()
                if args.a:
                    att_f = codecs.open("results/" + "attention_vis", "w", "utf-8")

                for i in range(len(X) // hp.batch_size):
                    ### Get mini-batches
                    x =              X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    x_length= X_length[i * hp.batch_size: (i + 1) * hp.batch_size]
                    y =              Y[i * hp.batch_size: (i + 1) * hp.batch_size]
                    y_tw =          TW[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources =  Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets =  Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    ppls = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds, ppl_step, att_ws, att_us, att_v = sess.run([g.preds, g.ppl, g.att_w, g.att_u, g.att_v],
                                                                           {g.x: x, g.x_length: x_length, g.y: y,
                                                                            g.y_tw: y_tw, g.y_decoder_input: preds})
                        preds[:, j] = _preds[:, j]
                        ppls[:, j] = ppl_step[:, j]

                    if args.a:
                        att_ws = np.mean(np.split(att_ws, hp.num_heads, axis=0), axis=0)  # (N, L, L)
                        att_us = np.mean(np.split(att_us, hp.num_heads, axis=0), axis=0)  # (N, T, T)
                        att_ws = np.reshape(att_ws, [hp.batch_size, hp.max_turn, hp.maxlen, hp.maxlen])
                        att_ws = np.mean(att_ws, axis=2)  # N, T, L
                        att_ws = np.reshape(att_ws, [hp.batch_size, hp.max_turn * hp.maxlen])
                        att_us = np.sum(att_us, axis=1)  # N, T

                    ### Write to file
                    for source, target, pred, ppl, att_w, att_u in zip(sources, targets, preds, ppls, att_ws, att_us): # sentence-wise
                        got = " ".join(idx2token[idx] for idx in pred).split("</S>")[0].strip()
                        if len(got.split()) > hp.gener_maxlen:
                            pred = pred.tolist()
                            pred_final = list(set(pred))
                            pred_final.sort(key=pred.index)
                        else:
                            pred_final = pred

                        got = " ".join(idx2token[idx] for idx in pred_final).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.write("- ppl_score: " + " ".join('%s' % np.mean(ppl)) + "\n\n")
                        if args.a:
                            att_f.write("- att_w: " + str(att_w) + "\n")
                            att_f.write("- att_u: " + str(att_u) + "\n\n\n\n\n")
                        fout.flush()
                        if args.a:
                            att_f.flush()

def parse_args():
    parser = argparse.ArgumentParser("evaluate_option")
    parser.add_argument("--a", action="store_true")
    parser.add_argument("--t", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    eval()
    print("Done")
    
    
