#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP_full_release

For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf

Authors
 * Yingzhi WANG 2021
"""

import os
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
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


class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
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
 
        # last dim will be used for AdaptativeAVG pool
        outputs = self.modules.slu_enc(layer_0)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs = self.modules.output_mlp(outputs)

        outputs = self.hparams.log_softmax(outputs)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        emoid, _ = batch.scenario_encoded

        """to meet the input form of nll loss"""
        emoid = emoid.squeeze(1)
        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec2_optimizer.step()
            self.optimizer.step()

        self.wav2vec2_optimizer.zero_grad()
        self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            torch.save(self.layers_weights, self.hparams.weights_path)

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_wav2vec2,
                new_lr_wav2vec2,
            ) = self.hparams.lr_annealing_wav2vec2(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec2_optimizer, new_lr_wav2vec2
            )

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "wave2vec_lr": old_lr_wav2vec2},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        print(self.layers_weights)
        self.wav2vec2_optimizer = self.hparams.wav2vec2_opt_class(
            [self.layers_weights]
        )
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.optimizer)
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec2_optimizer
            )
            self.checkpointer.add_recoverable("weights", self.layers_weights)



def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]


    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides("scenario", "scenario_encoded")
    def label_pipeline(semantics):
        scenario = semantics.split("'")[3]
        yield scenario
        scenario_encoded = label_encoder.encode_label_torch(scenario)
        yield scenario_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "scenario", "scenario_encoded"],
    )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets[0]],
        output_key="scenario",
    )
    
    return {"train": datasets[0], "valid" : datasets[1], "test" : datasets[2]} 


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    from transformers import AutoProcessor, AutoModelForPreTraining, Wav2Vec2FeatureExtractor
    from transformers import AutoModel



    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # freeze the feature extractor part when unfreezing

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    if not os.path.exists(hparams["weights_path"]): 
        #Zero initializiation like in S3PRL, not sure if it is the best choice, test torch.rand !   
        end_init = torch.cat([torch.zeros(hparams["num_layers"])])
    else : 
        end_init = torch.load(hparams["weights_path"])
        print("loaded weights")
    emo_id_brain.layers_weights = torch.nn.Parameter(end_init, requires_grad=True)
    torch.save(emo_id_brain.layers_weights, hparams["weights_path"])
    emo_id_brain.layers_weights.to(emo_id_brain.device)

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = emo_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )