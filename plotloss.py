import re
import matplotlib.pyplot as plt

def parse_log_and_plot(log_file_path):
    with open(log_file_path, 'r') as file:
        log_text = file.read()

    epoch_info = {}
    lines = log_text.split('\n')
    for line in lines:
        if "Epoch" in line:
            epoch = int(re.search(r'Epoch (\d+)', line).group(1))
            epoch_info[epoch] = {}
        elif "Train Loss" in line:
            train_loss = float(re.search(r'Train Loss: ([\d.]+)', line).group(1))
            epoch_info[epoch]["Train Loss"] = train_loss
        elif "Valid Loss" in line:
            valid_loss = float(re.search(r'Valid Loss: ([\d.]+)', line).group(1))
            epoch_info[epoch]["Valid Loss"] = valid_loss

    epochs = list(epoch_info.keys())
    train_losses = [epoch_info[epoch]["Train Loss"] for epoch in epochs]
    valid_losses = [epoch_info[epoch]["Valid Loss"] for epoch in epochs]

    plt.plot(epochs, train_losses, label='Train Loss')
    # plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence')
    plt.legend()
    plt.show()

log_file_path = 'train.log'  # 替换为实际的日志文件路径
parse_log_and_plot(log_file_path)