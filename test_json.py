train_save_path = '/home/csu-lyh/桌面/fine-tuning/train_data'

def test():
    import torch
    # 如果输出为 True，则表示支持 CUDA
    print(torch.cuda.is_available())
    # 显示检测到的 GPU 数量
    print(torch.cuda.device_count())
    print(torch.__version__)

    # 验证 json 是否能够正确解析
    import json
    train_data = json.load(open(f'{train_save_path}/train.json', 'r', encoding='utf8'))
    print(train_data[0])

    test_data = json.load(open(f'{train_save_path}/test.json', 'r',   encoding='utf8'))
    print(test_data[0])