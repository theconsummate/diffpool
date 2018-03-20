import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import time

import encoders
import gen.feat as featgen
import gen.data as datagen
import util


def synthetic_task_train(dataset, args, same_feat=True):

    model = encoders.GcnEncoderGraph(args.input_dim, args.hidden_dim, args.output_dim, 2)
    
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    times = []
    for batch_idx, data in enumerate(data):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1),
        average="micro"))
    print("Average batch time:", np.mean(times))


def synthetic_task1(args):

    # data
    graphs1 = datagen.gen_ba(range(40, 60), range(4, 5), 20, 
                             featgen.ConstFeatureGen(np.ones(args.input_dim)))
    for G in graphs1:
        G.graph['label'] = 0
    util.draw_graph_list(graphs1[:16], 4, 4, 'figs/ba')

    graphs2 = datagen.gen_2community_ba(range(20, 30), range(4, 5), 20, 0.3, 
                                        [featgen.ConstFeatureGen(np.ones(args.input_dim))])
    for G in graphs2:
        G.graph['label'] = 1
    util.draw_graph_list(graphs2[:16], 4, 4, 'figs/ba2')

    graphs = graphs1 + graphs2

    # minibatch
    dataset_sampler = GraphSampler(graphs)
    dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers)
    #synthetic_task_train(dataset, args)
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
            help='Output dimension')

    parser.set_defaults(dataset='',
                        feature_type='default',
                        lr=0.001,
                        batch_size=2,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=30,
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)

    synthetic_task1(args)

if __name__ == "__main__":
    main()
