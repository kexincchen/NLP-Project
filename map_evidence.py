import numpy as np
from utils import load_data

def extract_evidences(evidence_dict, evidence_keys):
    # Using list comprehension to extract values
    evidence_values = [
        evidence_dict[key] for key in evidence_keys if key in evidence_dict
    ]
    return evidence_values


def main():
    evidence_keys = np.load("result.npy")
    # evidence_data = load_data("data/evidence.json")

    # extracted_evidences = extract_evidences(evidence_data, evidence_keys)
    # print("Extracted Evidences:")
    print(evidence_keys.shape)
    # for evidence in extracted_evidences:
    #     print(evidence)


main()
