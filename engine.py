import torch.optim as optim
from util import *
from model import *
from sklearn.metrics import accuracy_score
torch.autograd.set_detect_anomaly(True)
def get_evaluation( y_prob, y_true):
    # accuracy = accuracy_score(y_true, y_prob)
    if np.argmax(y_prob) == np.argmax(y_true):
        return 1.0
    else:
        return 0.0


    print(y_prob, y_true)
    if y_prob[0] < 0.5:
        y_prob[0] = 0
    else:
        y_prob[0] = 1
    y_prob = np.array(y_prob)
    accuracy = accuracy_score(y_true, y_prob)
    return accuracy

class trainer():
    def __init__(self, max_depth, max_number_child, device, embed_dict,cgf_input_dim,cgf_bias, rst_input_dim, n_layers,
                 rst_bias, rst_drop_prob, rst_bidirect,cgf_drop_prob, cgf_bidirect, clip, lrate, wdecay):
        self.model = Net(max_depth, max_number_child, device, embed_dict, cgf_input_dim,cgf_bias, rst_input_dim,n_layers,
                         rst_bias, rst_drop_prob, rst_bidirect,cgf_drop_prob, cgf_bidirect)
        self.model.to(device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = nn.BCELoss()
        # self.loss = nn.CrossEntropyLoss()
        self.clip = clip
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lrate, weight_decay=wdecay)

    def train(self, train_root, train_bodytext, train_label):
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(train_root, train_bodytext)[0][0]

        loss = self.loss(output, train_label)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        acc = get_evaluation(output.tolist(), train_label.tolist())

        return [loss.tolist(), acc], (output.tolist(), train_label.tolist())

    def eval(self, root, bodytext, label):
        self.model.eval()
        output = self.model(root, bodytext)[0][0]

        loss = self.loss(output, label)

        acc = get_evaluation(output.tolist(), label.tolist())

        return [loss.tolist(), acc], (output.tolist(), label.tolist())
