from sklearn.metrics import accuracy_score, log_loss
from preparation import vocab
from decoder import generate_batch
from preparation import train_img_embeds, train_captions, train_captions_index, vocab_inverse
from hyperparameters import batch_size
from decoder import decoder
import os


def decode_sentence(sentence_indices):
    """
    decode the sentence from "vocab[index1], vocab[index2], ..." 
    to "v1, v2, ..."
    """
    return " ".join(list(map(vocab_inverse.get, sentence_indices)))

def check_after_training(n_examples):
    fd = generate_batch(train_img_embeds, train_captions_index, batch_size)
    logits = decoder.token_logits.eval(fd)
    truth = decoder.ground_truth.eval(fd)
    print("Loss:", decoder.loss.eval(fd))
    print("Accuracy:", accuracy_score(logits.argmax(axis =1), truth))
    for example_idx in range(n_examples):
        print("Example", example_idx)
        print("Predicted:", decode_sentence(logits.argmax(axis=1).reshape((batch_size, -1))[example_idx]))
        print("Truth:", decode_sentence(truth.reshape((batch_size, -1))[example_idx]))
        print("")

check_after_training(3)

