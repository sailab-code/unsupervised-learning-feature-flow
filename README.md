# Continual Learning of Conjugated Visual Representations through Higher-order Motion Flows (CMOSFET)

Learning with neural networks from a continuous stream of visual information introduces several challenges due to the non-i.i.d. nature of the data. 
However, it also opens to novel opportunities in the development of representations that are consistent with the information flow. 
In this work we investigate the case of unsupervised continual learning of pixel-wise features subject to multiple motion-induced constraints, therefore named _motion-conjugated representations_. 

Differently from existing approaches, motion is not a given signal (either ground-truth or estimated by external modules), but it is the outcome of a progressive and autonomous learning process, that takes place at different levels of the feature hierarchy. 
Multiple motion flows are estimated with neural networks and characterized by different levels of abstractions, ranging from the classic optical flow to other latent signals that originate from higher-level features and result in higher-order motions. 

Continuously learning to develop coherent multi-order flows and representations is prone to trivial solutions that we propose to avoid introducing a self-supervised contrastive loss, spatially-aware and based on flow-induced similarity. 

CODE REPOSITORY CONTENTS
------------------------
The datasets needed for reproducing experiments can be downloaded from the benchmark proposed by Tiezzi et al.,
which can be found at [this link](https://github.com/sailab-code/cl_stochastic_coherence)  in the `data` folder:
the content of the folder should be downloaded and inserted into the `data` folder of this repository. Similarly, for data of real-world videos, you can find prepared streams [here](https://drive.google.com/file/d/1gweNptwwgKzyrxsCxKw4lu9MgnEz_tMf/view?usp=sharing).

    confeat :               folder containing model definitions
    data :                  folder containing the rendered streams
    lve :                   source folder for handling the video processing and model creation and training  
    run_conj.py :           experiments runner
    run_conj_sparse.py :    experiments runner to be used when ground truth is provided in a small subset of frames (real world videos)
    best_cmosfet_runs.txt : command lines (and parameters) to reproduce the main results

REPRODUCE PAPER RESULTS
-----------------------

We tested our code with `PyTorch 1.10`. Please install the other required dependencies by running:

```
pip install -r requirements.txt
```

In the `best_cmosfet_runs.txt` file there are the command lines (hence, the experiments parameters) required to reproduce
the experiments of the main results (Table 1).

RUNNING EXPERIMENTS
-------------------
We provide Python scripts to easily test the proposed model (`run_conj.py` is to be used when ground truth is provided for every frame, `run_conj_sparse.py` is to be used when ground truth is provided in a small subset of frames). 
The PyTorch device is chosen through the `--device` argument (`cpu`, `cuda:0`,
`cuda:1`, etc.).

    usage: run_conj.py [-h] [--laps_unsup LAPS_UNSUP] [--laps_sup LAPS_SUP] [--laps_metrics LAPS_METRICS] 
                 [--step_size_features STEP_SIZE_FEATURES] [--step_size_displacements STEP_SIZE_DISPLACEMENTS] 
                 [--force_gray {yes,no}]  [--dataset DATASET] 
                 [--seed SEED] [--crops CROPS] [--flips FLIPS] [--jitters JITTERS]
                 [--lambda_c_lower LAMBDA_C_LOWER] [--lambda_c_upper LAMBDA_C_UPPER] [--lambda_r LAMBDA_R]  [--lambda_s LAMBDA_S] 
                 [--feature_planes FEATURE_PLANES]  [--device DEVICE] [--features_block FEATURES_BLOCK] [--displacement_block DISPLACEMENT_BLOCK]
                 [--wandb WANDB]
                 [--num_pairs NUM_PAIRS] [--lambda_sim LAMBDA_SIM]
                 [--similarity_threshold SIMILARITY_THRESHOLD] [--dissimilarity_threshold DISSIMILARITY_THRESHOLD] [--moving_threshold MOVING_THRESHOLD] 
                 [--sampling_type {plain,motion,features,motion_features}] [--kept_pairs_perc KEPT_PAIRS_PERC] 
                 [--simdis_loss_tau SIMDIS_LOSS_TAU] [--teacher {yes,no}] [--teacher_ema_weight TEACHER_EMA_WEIGHT] 
                 [--simdis_neg_avg SIMDIS_NEG_AVG]

Argument description/mapping with respect to the paper notation:

        --laps_unsup :  number of unsupervised laps where the coherence losses are minimized
        --laps_sup : number of laps on which the supervised templates are provided
        --laps_metrics : number of laps on which the metrics are computed (here the model weight are frozen, no learning is happening)
        --step_size_features : learning rate for the features branch
        --step_size_displacements : learning rate for the displacements branch
        --crops :  number of random crop augmentations
        --jitters :  number of jitters augmentations
        --flips :  number of horizontal flips  augmentations
        --lambda_c_lower : \lambda_{low} in the paper
        --lambda_c_upper : \lambda_{cur} in the paper
        --lambda_c_skip : \lambda_{skip} in the paper
        --lambda_r : \lambda_r in the paper
        --lambda_sim : \beta_f in the paper
        --lambda_s : \lambda_s in the paper 
        --simdis_loss_tau : \tau in the paper  
        --num_pairs: maximum number of points for the sampling procedure (corresponds to \eta * \eta from the paper)  
        --max_supervisions : number of supervisions per object
        --force_gray : if "yes", it corresponds to the "BW" of the paper. "no" requires an RGB stream
        --feature_planes : output dimension (number of features) of the features neural branch
        --displacement_block : architecture for the motion prediction branch [default: resunetblocknolastskip]
        --features_block : architecture for the feature branch [default: resunetblock_bias]
        --similarity_threshold : \tau_p in the paper
        --dissimilarity_threshold : \tau_n in the paper
        --moving_threshold : \tau_m in the paper
        --block_scheduling : T_{sched} in the paper
        --dataset  : specify the input stream
        --sampling_type : the nature of the sampling in {plain,motion,features,motion_features} (see ablation studies)
        --kept_pairs_perc : \aleph in the paper
        --teacher : activate the EMA net (default yes)
        --teacher_ema_weight  : \xi in the paper
        --seed : specify the seed


COLLECTING THE FINAL METRICS 
---------------------------

The final metrics are dumped in the `model_folder/final_metrics.json` file.

This file contains a dictionary with multiple metrics. The metrics that are reported in Table 1 of the paper are under
the key `f1_window_whole_global`. The F1-based on attention trajectory can be found under the key `f1_window_foa_global`.

COMPETITORS
-----------
For the competitors performances we re-runned [the code by Tiezzi et al](https://github.com/sailab-code/cl_stochastic_coherence).
We slightly updated the code to support the [MOCO](https://github.com/facebookresearch/moco) and 
[PixPro](https://github.com/zdaxie/PixPro) models (pre-trained weights can be found in the respective repositories).  


_NOTICE: PyTorch does not guarantee Reproducibility is not guaranteed by PyTorch across different releases, platforms, hardware. Moreover,
determinism cannot be enforced due to use of PyTorch operations for which deterministic implementations do not exist
(e.g. bilinear upsampling)._



