# ================================================
# Evaluating the Recurrent Neural Network (RNN)
# ================================================
import torch


import src.evaluate.evaluate as evaluate
import src.model.model as model
import src.preprocess.dataset as dataset

def evaluate(classifying_char_rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    classifying_char_rnn.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = classifying_char_rnn(text_tensor)
            guess, guess_i = model.label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing each row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    return confusion


confusion = evaluate(model.classifying_rnn, dataset.test_set,
                     classes=dataset.alldata.labels_uniq)
