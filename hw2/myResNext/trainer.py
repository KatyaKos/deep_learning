from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, model, optimize, loss):
        self.model = model
        self.optimize = optimize
        self.loss = loss
        self.log = SummaryWriter("log")

    def train(self, dataset, batch=10, epochs=5):
        data = DataLoader(dataset, batch_size=batch, shuffle=True)
        for i in range(epochs):
            for input, output in data:
                self.optimize.zero_grad()
                predicted = self.model(input)
                loss = self.loss(predicted, output)
                loss.backward()
                self.optimize.step()
            loss = self.statistics(data)
            self.log.add_scalar('Training loss', loss, i)

    def statistics(self, data):
        loss = 0.0
        for input, output in data:
            predicted = self.model(input)
            loss += self.loss(predicted, output).item()
        return loss / len(data)