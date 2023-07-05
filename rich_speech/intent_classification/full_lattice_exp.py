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
import numpy as np
import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import torch
import jsonlines

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        #normalized = self.asr_model.normalizer(wavs, 16000)
        lattice, lens = batch.encoded
        lattice = lattice.to(self.device)
        lattice= torch.exp(lattice)
        print(torch.sum(lattice[0][0].sum(axis=1))
        regrouped = self.module.regroup(lattice)
        lens = lens.to(self.device)
        # last dim will be used for AdaptativeAVG pool
        outputs = self.modules.slu_enc(regrouped)
        outputs = self.hparams.avg_pool(outputs, lens)
        outputs = outputs.view(outputs.shape[0], -1)
        outputs_scenario = self.modules.output_mlp(outputs)
        outputs_action = self.modules.output_mlp_actions(outputs)

        outputs_scenario = self.hparams.log_softmax(outputs_scenario)
        outputs_action = self.hparams.log_softmax(outputs_action)
        return [outputs_scenario, outputs_action]

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        scenario_id, _ = batch.scenario_encoded
        action_id, _ = batch.action_encoded
        predictions_scenario, predictions_actions = predictions
        """to meet the input form of nll loss"""
        scenario_id = scenario_id.squeeze(1)
        action_id = action_id.squeeze(1)
        loss = self.hparams.compute_cost(predictions_scenario, scenario_id) + self.hparams.compute_cost(predictions_actions, action_id)
        if stage != sb.Stage.TRAIN:
            self.scenario_error_metrics.append(batch.id, predictions_scenario, scenario_id)
            self.action_error_metrics.append(batch.id, predictions_actions, action_id)
        if stage == sb.Stage.TEST: 
            # write to "predictions.jsonl"
            scenario_predictions = torch.argmax(predictions_scenario, dim= 1) 
            action_predictions = torch.argmax(predictions_actions, dim=1)
            with jsonlines.open(
                hparams["output_folder"] + "/predictions.json", mode="a"
            ) as writer:
                for i in range(len(batch.id)):
                    dict_file = {}
                    dict_file["file"] = batch.id[i]
                    dict_file["action"] = action_predictions[i].item()

                    dict_file["true_action"] = action_id[i].item()
                    dict_file["true_scenario"] = scenario_id[i].item()
                    dict_file["scenario"] = scenario_predictions[i].item()
                    writer.write(dict_file)


        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()

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
            self.scenario_error_metrics = self.hparams.scenario_error_stats()
            self.action_error_metrics = self.hparams.action_error_stats()

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

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "scenario_error_rate": self.scenario_error_metrics.summarize("average"),
                "action_error_rate": self.action_error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["scenario_error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

             # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats, min_keys=["scenario_error_rate"]
            )

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.optimizer)


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
    @sb.utils.data_pipeline.provides("encoded")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        file_name = wav.split("/")[-1] +".npy"
        file_path = os.path.join(hparams["encoded_folder"], file_name)
        encoded = torch.Tensor(np.load(file_path))
        encoded = torch.squeeze(encoded)
        return encoded

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder_scenarios = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_actions = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides("scenario", "scenario_encoded", "action", "action_encoded")
    def label_pipeline(semantics):
        scenario = semantics.split("'")[3]
        yield scenario
        scenario_encoded = label_encoder_scenarios.encode_label_torch(scenario)
        yield scenario_encoded
        action = semantics.split("'")[7]
        yield action
        action_encoded = label_encoder_actions.encode_label_torch(action)
        yield action_encoded


    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "encoded", "scenario", "scenario_encoded", "action", "action_encoded"],
    )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file_scenarios = os.path.join(hparams["save_folder"], "label_encoder_scenarios.txt")
    lab_enc_file_actions = os.path.join(hparams["save_folder"], "label_encoder_actions.txt")
    label_encoder_scenarios.load_or_create(
        path=lab_enc_file_scenarios,
        from_didatasets=[datasets[0]],
        output_key="scenario",
    )
    label_encoder_actions.load_or_create(
        path=lab_enc_file_actions,
        from_didatasets=[datasets[0]],
        output_key="action",
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
    encoder_path= "/gpfsscratch/rech/nou/uzn19yk/encoder_outputs/"
    # Data preparation, to be run on only one process.
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)


    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

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
        min_key="scenario_error_rate",
        test_loader_kwargs=hparams["dataloader_options"],
    )
