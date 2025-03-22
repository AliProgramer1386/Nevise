import os
import torch
from hazm import Normalizer
from models import SubwordBert
from helpers import load_vocab_dict
from utils import get_sentences_splitters
from helpers import bert_tokenize_for_valid_examples, labelize, untokenize_without_unks, untokenize_without_unks2

def load_spell_checker(vocab_path="model/vocab.pkl", model_checkpoint_path="model/model.pth.tar"):
    """
    Loads the spell checker model, vocabulary, and normalizer.

    Args:
        vocab_path (str): Path to the vocabulary file.
        model_checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        A tuple containing the model, vocabulary, device, and normalizer.
    """
    # Set device (GPU if available, otherwise CPU)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load vocabulary
    print(f"Loading vocab from {vocab_path}")
    vocab = load_vocab_dict(vocab_path)
    
    # Load model
    model = SubwordBert(3 * len(vocab["chartoken2idx"]), vocab["token2idx"][vocab["pad_token"]], len(vocab["token_freq"]))
    model = load_pretrained(model, model_checkpoint_path)
    model.to(DEVICE)
    
    # Load normalizer
    normalizer = Normalizer()
    
    return model, vocab, DEVICE, normalizer


def correct_three_texts(model, vocab, device, normalizer, text1, text2, text3):
    """
    Corrects the spelling of three input texts.

    Args:
        model: The spell checking model.
        vocab: The vocabulary used by the model.
        device: The device (CPU or GPU) to run the model on.
        normalizer: The text normalizer.
        text1, text2, text3: The three input texts to correct.

    Returns:
        A tuple containing the corrected versions of the three input texts.
    """
    def spell_checking_on_sents(model, vocab, device, normalizer, txt):
        """
        Corrects the spelling of a single text.
        """
        sents, splitters = get_sentences_splitters(txt)
        sents = [utils.space_special_chars(s) for s in sents]
        sents = list(filter(lambda txt: (txt != '' and txt != ' '), sents))
        test_data = [(normalizer.normalize(t), normalizer.normalize(t)) for t in sents]
        greedy_results = model_inference(model, test_data, topk=1, DEVICE=device, BATCH_SIZE=1, vocab_=vocab)
        corrected_text = ''.join([line['predicted'] + splitter for line, splitter in zip(greedy_results, splitters + ['\n'])])
        return corrected_text

    # Correct each text
    corrected_text1 = spell_checking_on_sents(model, vocab, device, normalizer, text1)
    corrected_text2 = spell_checking_on_sents(model, vocab, device, normalizer, text2)
    corrected_text3 = spell_checking_on_sents(model, vocab, device, normalizer, text3)
    
    return corrected_text1, corrected_text2, corrected_text3


def model_inference(model, data, topk, DEVICE, BATCH_SIZE=16, vocab_=None):
    """
    Runs inference on the model for a given batch of data.
    """
    if vocab_ is not None:
        vocab = vocab_
    print("###############################################")
    inference_st_time = time.time()
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    VALID_BATCH_SIZE = BATCH_SIZE
    valid_loss = 0.
    print("data size: {}".format(len(data)))
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    results = []
    line_index = 0
    for batch_id, (batch_labels, batch_sentences) in tqdm(enumerate(data_iter)):
        torch.cuda.empty_cache()
        st_time = time.time()
        # set batch data for bert
        batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits = bert_tokenize_for_valid_examples(batch_labels, batch_sentences)
        if len(batch_labels_) == 0:
            print("################")
            print("Not predicting the following lines due to pre-processing mismatch: \n")
            print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
            print("################")
            continue
        else:
            batch_labels, batch_sentences = batch_labels_, batch_sentences_
        batch_bert_inp = {k: v.to(DEVICE) for k, v in batch_bert_inp.items()}
        # set batch data for others
        batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels_ids = batch_labels_ids.to(DEVICE)

        try:
            with torch.no_grad():
                """
                NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
                """
                batch_loss, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids, topk=topk)
        except RuntimeError:
            print(f"batch_bert_inp:{len(batch_bert_inp.keys())},batch_labels_ids:{batch_labels_ids.shape}")
            raise Exception("")
        valid_loss += batch_loss
        batch_lengths = batch_lengths.cpu().detach().numpy()
        if topk == 1:
            batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_sentences)
        else:
            batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_sentences, topk=None)
        batch_clean_sentences = [line for line in batch_labels]
        batch_corrupt_sentences = [line for line in batch_sentences]
        batch_predictions = [line for line in batch_predictions]

        for i, (a, b, c) in enumerate(zip(batch_clean_sentences, batch_corrupt_sentences, batch_predictions)):
            results.append({"id": line_index + i, "original": a, "noised": b, "predicted": c, "topk": [], "topk_prediction_probs": [], "topk_reranker_losses": []})
        line_index += len(batch_clean_sentences)

    print(f"\nEpoch {None} valid_loss: {valid_loss / (batch_id + 1)}")
    print("total inference time for this data is: {:4f} secs".format(time.time() - inference_st_time))
    print("###############################################")
    return results


def load_pretrained(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Loads a pretrained model from a checkpoint.
    """
    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    print(f"Loading model params from checkpoint dir: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    max_dev_acc, argmax_dev_acc = checkpoint_data["max_dev_acc"], checkpoint_data["argmax_dev_acc"]

    if optimizer is not None:
        return model, optimizer, max_dev_acc, argmax_dev_acc
    return model


if __name__ == '__main__':
    # Example usage of the main function (optional)
    pass
