#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_with_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

Authors
 * Sung-Lin Yeh 2021
 * Titouan Parcollet 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch.nn.functional as F
import s3prl
from s3prl import hub
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.upstream.wav2vec2.wav2vec2_model import (
    AudioPretrainingConfig,
    Wav2Vec2Config,
    Wav2Vec2Model,
)
from s3prl.nn.upstream import Featurizer, S3PRLUpstream, UpstreamDownstreamModel
filein = "/gpfsstore/rech/nou/uzn19yk/wav2vec_vox_new.pt"
build_upstream = {'name': 'wav2vec2_large_ll60k'}
build_featurizer = {'layer_selections': None, 'normalize': True}

s3prl_str = "wav2vec2_large_ll60k"

w_s3prl = getattr(hub, s3prl_str)().cuda()
w_s3prl.wav_normalize = True
w_s3prl.apply_padding_mask = False
w_s3prl.eval()

#upstream = S3PRLUpstream(**build_upstream)

from pyctcdecode import build_ctcdecoder

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Forward pass

        hiddens= w_s3prl(wavs)["hidden_states"]
        #new_feats= hf_model(wavs)
        x= hiddens
        new_features = [F.layer_norm(t, (t.shape[-1],)) for t in x]
        x=new_features

        #print(x.shape) #First dimension should be equal to the number of layers in the hparams
        norm_weights = torch.nn.functional.softmax(self.layers_weights, dim=-1)
        layer_0 =(x[0].detach()) * norm_weights[0]
        for i in range(1, len(x)): 
            layer_0 += (x[i].detach()) * norm_weights[i]
        y = self.modules.enc(layer_0)
        y=y[0]
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(y)
        p_ctc = self.hparams.log_softmax(logits)
        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc
        if stage ==sb.Stage.VALID:
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)


        if stage == sb.Stage.TEST:
            # Decode token terms to words
            predicted_words =[]
            for logs in p_ctc: 
                text = decoder.decode(logs.detach().cpu().numpy())
                predicted_words.append(text.split(" "))

            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)


        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.hparams.model.parameters(), 1.0)
        if self.check_gradients(loss):
            self.model_optimizer.step()
            self.weights_optimizer.step()

        self.model_optimizer.zero_grad()
        self.weights_optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_weights, new_lr_weights= self.hparams.lr_annealing_weights(
                stage_stats["loss"]
            )
            torch.save(self.layers_weights, self.hparams.weights_path)
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.weights_optimizer, new_lr_weights
            )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the weights optimizer and model optimizer"
        self.layer_norms =[]
        self.layer_norms_params = []
        for i in range(self.hparams.num_layers):
            self.layer_norms.append(torch.nn.LayerNorm(1024, elementwise_affine=False).to(self.device))
        self.weights_optimizer = self.hparams.weights_opt_class(
            [self.layers_weights]
        )
        self.model_optimizer = self.hparams.model_opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            self.checkpointer.add_recoverable(
                "weights_opt", self.weights_optimizer
            )



def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "char_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets, label_encoder


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Loading the SSL model from huggingface 
    # Important point here, "output_hidden_states: true " should be added in the config.json file 
    #hf_model = AutoModelForPreTraining.from_pretrained(hparams["wav2vec2_hub"])
    #hf_model = AutoModel.from_pretrained(hparams["wav2vec2_hub"])
    #hf_model.eval()
    # Dataset prep (parsing Librispeech)
    def read_labels_file(labels_file): 
        with open(labels_file, "r") as lf: 
            lines = lf.read().splitlines()
            division = "==="
            numbers = {}
            for line in lines : 
                if division in line : 
                    break
                string, number = line.split("=>")
                number = int(number)
                string = string[1:-2]
                numbers[number] = string
            return [numbers[x] for x in range(len(numbers))]

    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    """
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )
    """

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    #hf_model.to(asr_brain.device)
    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder
    if not os.path.exists(hparams["weights_path"]): 
        #Zero initializiation like in S3PRL, not sure if it is the best choice, TODO TODO test torch.rand !   
        end_init = torch.cat([torch.zeros(hparams["num_layers"])])
    else : 
        end_init = torch.load(hparams["weights_path"])
        print("loaded weights")
    asr_brain.layers_weights = torch.nn.Parameter(end_init, requires_grad=True)
    torch.save(asr_brain.layers_weights, hparams["weights_path"])
    asr_brain.layers_weights.to(asr_brain.device)
    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
    labels = read_labels_file(os.path.join(hparams["save_folder"], "label_encoder.txt"))
    print(labels)
    labels = [""] + labels[1:]
    print(len(labels))
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path="/gpfsstore/rech/nou/uzn19yk/4-gram.arpa",  # either .arpa or .bin file
        alpha=0.5,  # tuned on a val set
        beta=1.0,  # tuned on a val set
    )


    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
