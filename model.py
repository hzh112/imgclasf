import numpy as np
import tensorflow as tf

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 归一化数据
train_images = train_images.reshape((train_images.shape[0], 784)).astype('float32') / 255.0
test_images = test_images.reshape((test_images.shape[0], 784)).astype('float32') / 255.0

# 转换标签为独热编码
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # 使用ReLU激活函数
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)  # 使用softmax激活函数
        return self.a2

    def compute_loss(self, predictions, targets):
        m = predictions.shape[0]
        return -np.sum(targets * np.log(predictions + 1e-15)) / m

    def backward(self, x, y):
        m = x.shape[0]

        # 计算损失梯度
        d_loss = self.a2 - y

        # 反向传播到第二层
        d_weights2 = np.dot(self.a1.T, d_loss)
        d_bias2 = np.sum(d_loss, axis=0, keepdims=True)

        # 反向传播到第一层
        d_hidden = np.dot(d_loss, self.weights2.T)
        d_hidden[self.z1 <= 0] = 0  # ReLU的反向传播

        d_weights1 = np.dot(x.T, d_hidden)
        d_bias1 = np.sum(d_hidden, axis=0, keepdims=True)

        return d_weights1, d_bias1, d_weights2, d_bias2

    def update_params(self, d_weights1, d_bias1, d_weights2, d_bias2, learning_rate):
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
def compute_accuracy(predictions, targets):
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
epochs = 10
batch_size = 32

model = SimpleNN(input_size, hidden_size, output_size)

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, len(train_images), batch_size):
        batch_x = train_images[i:i + batch_size]
        batch_y = train_labels[i:i + batch_size]

        # 前向传播
        predictions = model.forward(batch_x)

        # 计算损失
        loss = model.compute_loss(predictions, batch_y)
        epoch_loss += loss

        # 反向传播
        d_weights1, d_bias1, d_weights2, d_bias2 = model.backward(batch_x, batch_y)

        # 更新参数
        model.update_params(d_weights1, d_bias1, d_weights2, d_bias2, learning_rate)

    # 输出每个epoch的平均损失
    avg_loss = epoch_loss / (len(train_images) // batch_size)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

    # 计算并输出训练集的准确率
    train_predictions = model.forward(train_images)
    train_accuracy = compute_accuracy(train_predictions, train_labels)
    print(f"Epoch {epoch + 1}, Training Accuracy: {train_accuracy * 100:.2f}%")

    # 计算并输出测试集的准确率
    test_predictions = model.forward(test_images)
    test_accuracy = compute_accuracy(test_predictions, test_labels)
    print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy * 100:.2f}%")


