# NLP-Project

## Experiment 

### BAttn + 2 layers of LSTM
Test loss 0.13740038509879793
Score of LSTM {'precision': 0.7376237623762376, 'recall': 0.739454094292804, 'fscore': 0.7385377942998761}

> python eval.py --predictions data/output/BAttn-lstm2-t70-output.json --groundtruth data/dev-claims.json
Evidence Retrieval F-score (F)    = 0.00043868980374046364
Claim Classification Accuracy (A) = 0.44155844155844154
Harmonic Mean of F and A          = 0.0008765087930332735

### BAttn + 2 layers of Bidirectional-LSTM
> python eval.py --predictions data/output/BAttn-blstm-t70-output.json --groundtruth data/dev-claims.json
Evidence Retrieval F-score (F)    = 0.0005359535095910537
Claim Classification Accuracy (A) = 0.44155844155844154
Harmonic Mean of F and A          = 0.0010706075403031736