import time
import torch
from ..logger import Logger
from yacs.config import CfgNode

def train_HGNN(loader, model, optimizer, scheduler, logger: Logger, cfg: CfgNode) -> None:
    loader.split(type='Hetero')
    # loader.Embedding_hetero()
    graph_hete = loader.hetero
    hetero_mask = loader.hetero_mask
    hete_class = loader.hetero_y

    start = time.time()
    best = -1e8
    for epoch in range(cfg.train.epoch):
        model.train()
        optimizer.zero_grad()
        output = model(graph_hete)
        train_mask = hetero_mask['train']
        target = hete_class[train_mask].squeeze()
        result = {}
        if cfg.dataset.task == 'classification':
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output[train_mask], target.to(torch.long))
            with torch.no_grad():
                preds = output[train_mask].argmax(dim=1)
                correct = (preds == target.to(torch.long)).sum().item()
                total = train_mask.shape[0]
                accuracy = correct / total
                result["train"] = accuracy
        elif cfg.dataset.task == 'regression':
            criterion = torch.nn.MSELoss()
            loss = criterion(output[train_mask].squeeze(), target.to(torch.float))
        else:
            raise ValueError("task must be either classification or regression.")
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for split in ['val', 'test']:
                mask = hetero_mask[split].to(torch.long)
                output = model(graph_hete)
                target = hete_class[mask].squeeze()
                if cfg.dataset.task == 'classification':
                    preds = output[mask].argmax(dim=1)
                    correct = (preds == target.to(torch.long)).sum().item()
                    total = mask.shape[0]
                    accuracy = correct / total
                    print(f'{split} Accuracy: {accuracy:.2%}')
                    result[split] = accuracy
                # elif args.task == 'regression':
                #     preds = output[mask].squeeze()
                #     RMSE = (preds - target).square().mean().sqrt()
                #     StdE = (target - target.mean()).square().mean().sqrt()
                #     result.append(-RMSE)
                #     log.append(f'{split} RMSE (Root Mean Square Error): {RMSE}, where Std Error: {StdE}')
                #     print(log[-1])
        logger.log_scalars("Accuracy", result, epoch)
        logger.log_scalar("Loss", loss.item(), epoch)
        logger.log_scalar("Time used", time.time() - start, epoch)
        # result, log = test(model, dataset, args)
        # if result[0] > best:
        #     best = result[0]
        #     logs[0] = 'Best ' + logs[0]
        #     logg = logs + log
        #     patc = 0
        # else:
        #     patc += 1
        #     if patc > args.patience:
        #         break
    return 
