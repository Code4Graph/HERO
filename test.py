import time
import argparse
from engine import *
import random
from util import *
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score

print("torch version :", torch.__version__)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--if_embedding', default=True, type=bool, metavar='N', help='embedding')
parser.add_argument('--cgf_input_dim', default=100, type=int, metavar='N', help='cgf_tree input dim/ word embedding dim')
parser.add_argument('--cgf_bias', default=False, type=bool, metavar='N', help='For lstm: If False, then the layer does not use bias weights b_ih and b_hh. Default: True')
parser.add_argument('--rst_input_dim', default=100, type=int, metavar='N', help='rst_tree input dim')
parser.add_argument('--n_layers', default=1, type=int, metavar='N', help='number of output classes')
parser.add_argument('--rst_bias', default=False, type=bool, metavar='N', help='For lstm: If False, then the layer does not use bias weights b_ih and b_hh. Default: True')
parser.add_argument('--rst_drop_prob', default=0.0, type=float, metavar='N', help='dropout for rst')
parser.add_argument('--rst_bidirect', default=True, type=bool, metavar='N', help='number of output classes')
parser.add_argument('--cgf_drop_prob', default=0.0, type=float, metavar='N', help='dropout for cgf')
parser.add_argument('--cgf_bidirect', default=True, type=bool, metavar='N', help='number of output classes')
parser.add_argument('--classes', default=2, type=int, metavar='N', help='number of output classes')
parser.add_argument('--max_depth', default=100, type=int, metavar='N', help='max depth for the tree')
parser.add_argument('--max_child', default=20, type=int, metavar='N', help='max child # for the node in cgf')
parser.add_argument('--clip', default=2, type=int, metavar='N', help='')
parser.add_argument('--print_every',type=int,default=10,help='')
parser.add_argument('--save', type = str, default="result/", help='')
parser.add_argument('--model_path', type = str, default='valid_auc_epoch_27_0.865.pth', help='')

args = parser.parse_args()

def pre_label_helper(pre, label):
    b = 0 if label[0] == 1 else 1
    a = 0 if pre[0] > pre[1] else 1

    return a, b

def main():

    device_setting = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device_setting)


    # print("===> reading relations ...")
    # cfg_relation = {}
    # path = 'POS_Categories.csv'
    # read_relation(path, cfg_relation)
    # rst_relation = {"NN":0, "NS":1, "SN":2}


    print("===> loading  embed ...")
    embed = {}
    f = open('glove.6B.100d.txt', encoding="utf8")
    for i in f:
        value = i.split()
        word = value[0]
        embed[word] = np.asarray( value[1:], dtype='float32')
    f.close()
    embeding_example = embed['the']
    print("embeding_example: ", embeding_example)
    print(type(embeding_example))

    #########################################################################
    # create model
    print("===> creating rnn model ...")
    model = Net(max_depth=args.max_depth, max_number_child=args.max_child, device=device_setting,
                embed_dict=embed, cgf_input_dim=args.cgf_input_dim,
                cgf_bias=args.cgf_bias, rst_input_dim=args.rst_input_dim,n_layers=args.n_layers,
                rst_bias=args.rst_bias, rst_drop_prob=args.rst_drop_prob, rst_bidirect=args.rst_bidirect,
                cgf_drop_prob=args.cgf_drop_prob, cgf_bidirect=args.cgf_bidirect)

    model.to(device_setting)
    print("args:", args)

    print("====> loading " + str(args.model_path) + "model to test!")
    retrain_model_path = args.model_path
    model.load_state_dict(torch.load(args.save + retrain_model_path))
    model.eval()

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    ##################################################################################
    # loading dataset
    print("===> creating dataloaders ...")
    end = time.time()
    data_loader = TextClassDataLoader('../data', batch_size=2)
    print('===> dataloader creatin: {t:.3f}'.format(t=time.time() - end))
    test_root = data_loader[0]
    test_bodytext = data_loader[1]
    test_label = data_loader[2]
    from collections import Counter
    print(Counter(test_label))
    print("test size: " , len(test_root))


    #######################################################################
    #test training model
    ttt_pre = []
    ttt_label = []
    test_idx = np.arange(0, len(test_root)).tolist()
    random.shuffle(test_idx)
    s1 = time.time()
    for iter, idx in enumerate(test_idx):
        tt_root = test_root[idx]
        tt_bodytext = test_bodytext[idx]
        if test_label[idx] == 0:
            tt_label = torch.Tensor([1, 0]).to(device_setting)
        else:
            tt_label = torch.Tensor([0, 1]).to(device_setting)
        with torch.no_grad():
            preds = model(tt_root, tt_bodytext)[0][0]

        preds = preds.tolist()
        label = tt_label.tolist()
        pre, label = pre_label_helper(preds, label)

        ttt_label.append(label)
        ttt_pre.append(pre)

    s2 = time.time()
    log = 'Test roc_auc_score_weighted: {:.4f}\n'

    f1 = f1_score(np.array(ttt_label), np.array(ttt_pre), average='micro')
    recall = recall_score(np.array(ttt_label), np.array(ttt_pre), average=None)
    precision = precision_score(np.array(ttt_label), np.array(ttt_pre), average=None)

    print(log.format(roc_auc_score(ttt_label, ttt_pre, average="weighted")), flush=True)
    print('f1: ', f1)
    print('recall: ', recall)
    print('precesion: ', precision)



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))




