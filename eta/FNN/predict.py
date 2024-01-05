import torch
import train_model
import pandas as pd

MODEL_FILE_PATH = "./time_predict_model.pth"


def predict(sigle_row_data):
    """
    X为单行DataFrame,格式参考train.csv
    """
    X, y = train_model.transfer_data(data=sigle_row_data)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 关闭梯度计算以节省内存和提高速度
    model = train_model.FeedForwardNN(train_model.INPUT_SIZE
                                      , train_model.HIDDEN_SIZES
                                      , train_model.OUTPUT_SIZE)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    with torch.no_grad():
        # 使用模型进行预测
        prediction = model(X_tensor)
        
    # print("model prediction:" + str(prediction[0][0]))
    # print("ground truth:" + str(y_tensor[0]))
    return prediction[0][0]