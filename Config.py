import os

class config:
    # 根目录
    root_path = os.getcwd()
    data_dir = os.path.join(root_path, './data/data/')
    train_data_path = os.path.join(root_path, 'data/train.json')
    test_data_path = os.path.join(root_path, 'data/test.json')
    output_path = os.path.join(root_path, './output')
    output_test_path = os.path.join(output_path, 'test.txt')
    load_model_path = None

    # 一般超参
    epoch = 40
    learning_rate = 3e-5
    weight_decay = 0
    num_labels = 5
    loss_weight = [1.68, 9.3, 3.36]

    # Fuse相关
    fuse_model_type = 'NaiveCombine'
    only = None
    middle_hidden_size = 64
    attention_nhead = 8
    attention_dropout = 0.4
    fuse_dropout = 0.5
    out_hidden_size = 128

    fixed_text_model_params = False

    # Dataloader params
    checkout_params = {'batch_size': 4, 'shuffle': False}
    train_params = {'batch_size': 16, 'shuffle': True, 'num_workers': 2}
    val_params = {'batch_size': 16, 'shuffle': False, 'num_workers': 2}
    test_params =  {'batch_size': 8, 'shuffle': False, 'num_workers': 2}

    
    