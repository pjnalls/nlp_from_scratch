# ================================================
# Training the Recurrent Neural Network (RNN)
# ================================================
import matplotlib.pyplot as plt
import random
import numpy as np
import time

import torch
import torch.nn as nn

import src.model.model as model
import src.preprocess.dataset as dataset


def train(rnn, training_data, n_epoch=10, n_batch_size=64, report_every=50, learning_rate=0.2, criterion=nn.NLLLoss()):
    """ Learn on a batch of training_data for a specified 
    number of iterations and reporting thresholds """
    # Keep track of losses
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad()  # Clear gradients

        # Create some mini-batches
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches))

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch:  # For each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # Optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches))
        if iter % report_every == 0:
            print(
                f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses


start = time.time()
all_losses = train(model.classifying_rnn, dataset.train_set,
                  n_epoch=27, learning_rate=0.15, report_every=5)
end = time.time()
print(f"training took {end - start}s")

torch.save(model.classifying_rnn, "models/model.pth")
# plt.figure()
# plt.plot(all_losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()
