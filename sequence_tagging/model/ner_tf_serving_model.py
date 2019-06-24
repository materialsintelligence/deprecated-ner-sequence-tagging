import numpy as np
np.random.seed(1)
import os
import requests
import tensorflow as tf
tf.set_random_seed(1)

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from .ner_model import NERModel

class NERServingModel(NERModel):
    """A variant of ner_model suitable for constructing and using a tf-serving API"""

    def __init__(self, config, api_url):
        super(NERServingModel, self).__init__(config)
        self.api_url = api_url


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self._api_call_predict(fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            raise Exception

    def _api_call_predict(self,feed_dict):
        """ Make a call to a tf-serving server implementing the NER model

        Args:
            feed_dict: dictionary of inputs

        Returns:
            logits, trans_params
        """

        r = requests.post(url=self.api_url, data=feed_dict)
        if r.status_code == 200:
            r = r.json()
            return r['logits'],r['trans_params']
        else:
            raise Exception

    def save_prediction_model(self,save_dir):
        """Makes a tf saved_model copy of the current NER model

        Args:
            save_dir: Directory to save the model

        Returns:
            self

        """

        tf.saved_model.simple_save(
        self.sess,
        save_dir,
        {"word_ids": self.word_ids,"sequence_lengths": self.sequence_lengths,"dropout":self.dropout},
        {"logits": self.logits,"trans_params":self.trans_params}
        )
        return self