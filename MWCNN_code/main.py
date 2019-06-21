import torch

import utility
import data
import model
import loss

# import h5py
from option import args
from trainer import Trainer



torch.set_num_threads(12)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:

    # args.model = 'NL_EST'
    # model1 = model.Model(args, checkpoint)
    #
    # args.model = 'KERNEL_EST'
    # model2 = model.Model(args, checkpoint)
    # args.model = 'BSR'
    model = model.Model(args, checkpoint)


    loader = data.Data(args)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

