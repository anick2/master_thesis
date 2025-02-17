# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
import os
import config as cfg

from model import ABAE
from reader import get_centroids, get_w2v, read_data_tensors

logger = logging.getLogger(__name__)


def main():
    print('started')
    w2v_model = get_w2v(cfg.word2vec_path)
    wv_dim = w2v_model.vector_size
    y = torch.zeros((cfg.batch_size, 1))

    model = ABAE(wv_dim=wv_dim,
                 asp_count=cfg.aspects_number,
                 init_aspects_matrix=get_centroids(w2v_model, aspects_count=cfg.aspects_number))
    logger.debug(str(model))

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters())

    for t in range(cfg.epochs):

        logger.debug("Epoch %d/%d" % (t + 1, cfg.epochs))

        data_iterator = read_data_tensors(cfg.data_path, cfg.word2vec_path, batch_size=cfg.batch_size, maxlen=cfg.max_len)

        for item_number, (x, texts) in enumerate(data_iterator):

            if x.shape[0] < cfg.batch_size:  # pad with 0 if smaller than batch size
                x = np.pad(x, ((0, cfg.batch_size - x.shape[0]), (0, 0), (0, 0)))

            x = torch.from_numpy(x)

            # extracting bad samples from the very same batch; not sure if this is OK, so todo
            negative_samples = torch.stack(
                tuple([x[torch.randperm(x.shape[0])[:cfg.negative_samples]]
                       for _ in range(cfg.batch_size)]))

            # prediction
            y_pred = model(x, negative_samples)

            # error computation
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if item_number % 100 == 0:

                print("%d batches, and LR: %.5f" % (item_number, optimizer.param_groups[0]['lr']))

                for i, aspect in enumerate(model.get_aspect_words(w2v_model, logger)):
                    print("[%d] %s" % (i + 1, " ".join([a for a in aspect])))

                print("Loss: %.4f" % loss.item())

                try:
                    torch.save(model, f"abae_%.2f_%06d.bin" % (loss.item(), item_number))
                except Exception as e:
                    print("Model saving failed.")


if __name__ == "__main__":
    main()
