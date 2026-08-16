# DT2F-TLNet

Implementation of the research paper:

**DT2F-TLNet: A Novel Text-Independent Writer Identification and Verification Model using Deep Type-2 Fuzzy Architecture and Transfer Learning Networks**

This project implements a Deep Type-2 Fuzzy Transfer Learning Network (DT2F-TLNet) for offline handwriting/signature verification.

### Approach
- Type-2 Fuzzy CNN for feature extraction
- InceptionV3 for transfer learning
- Feature fusion from both branches
- Binary classification of genuine and forged samples

### Dataset
CEDAR Signature Dataset.

### Results
- Test Accuracy: **81.44%**
- Precision: **77.40%**
- Recall: **85.61%**
- F1-Score: **81.29%**
- AUC: **0.8542**
