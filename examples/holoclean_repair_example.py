import holoclean
from detect import NullDetector, ViolationDetector
from repair.featurize import *


# 1. Setup a HoloClean session.
hc = holoclean.HoloClean(
    pruning_topk=0.0,
    epochs=10,
    weight_decay=0.1,
    threads=1,
    batch_size=1,
    verbose=True,
    timeout=3*60000,
    print_fw=True
).session

# 2. Load training data and denial constraints.
hc.load_data('hospital', '../testdata/hospital.csv')
hc.load_dcs('../testdata/hospital_constraints_att.txt')
hc.ds.set_constraints(hc.get_dcs())

# 3. Detect erroneous cells using these two detectors.
detectors = [NullDetector(), ViolationDetector()]
hc.detect_errors(detectors)

# 4. Repair errors utilizing the defined features.
hc.setup_domain()
featurizers = [
    InitAttFeaturizer(),
    InitSimFeaturizer(),
    OccurAttrFeaturizer(),
    FreqFeaturizer(),
    ConstraintFeat(),
    LangModelFeat()
]
hc.repair_errors(featurizers)


# 5. Evaluate the correctness of the results.
hc.evaluate('../testdata/hospital_clean.csv', 'tid', 'attribute', 'correct_val')
