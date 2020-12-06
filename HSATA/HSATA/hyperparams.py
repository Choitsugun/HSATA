class Hyperparams:
    '''Hyperparameters'''
    # data
    source_target_vocab = '../cop/train_data/corpra_dev_test_final_turn_len50_turn15.txt'
    topic_vocab = '../cop/train_data/corpra_dev_test_final_tp.txt'


    source_train = '../cop/train_data/corpra_dev_test_final_s.txt'
    target_train = '../cop/train_data/corpra_dev_test_final_t.txt'
    topic_train = '../cop/train_data/corpra_dev_test_final_tp.txt'

    #source_train = '../cop/test_data_128_25/corpra_dev_test_final_s.txt'
    #target_train = '../cop/test_data_128_25/corpra_dev_test_final_t.txt'
    #topic_train = '../cop/test_data_128_25/corpra_dev_test_final_tp.txt'


    source_test = '../cop/test_data/corpra_dev_test_final_s.txt'
    target_test = '../cop/test_data/corpra_dev_test_final_t.txt'
    topic_test = '../cop/test_data/corpra_dev_test_final_tp.txt'


    # training
    batch_size = 64     # alias = N
    lr = 0.0001         # learning rate. In paper, learning rate is adjusted to the global step.
    warmup_steps = 4000
    num_epochs = 50
    logdir = 'Cho'      # log directory
    
    # model
    maxlen = 51         # Maximum number of words in a sentence.
    max_turn=10         # Maximum number of turns of each context.
    tw_maxlen = 50      # Maximum number of topic words of each context.
    gener_maxlen = 40   # Maximum number of generated response words.
    min_cnt = 1         # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks_w = 4    # number of encoder/decoder blocks on word level
    num_blocks_u = 4    # number of encoder/decoder blocks on utterance level
    num_blocks = 4      # number of decoder blocks
    num_heads = 8       # number of self-attention head
    dropout_rate = 0.1
    penalty = 0.05