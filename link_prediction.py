import numpy as np
from gensim.models import KeyedVectors

from kbc_rdf2vec.dataset import DataSet
from kbc_rdf2vec.prediction import (
    PredictionFunctionEnum,
    RandomPredictionFunction,
    AveragePredicateAdditionPredictionFunction,
)

print('load embeddings')
kv = KeyedVectors.load_word2vec_format("embeddings/transr_entity.bin", binary=True)
for function in PredictionFunctionEnum:
    print(function)

    # forbid reflexive
    function_instance = function.get_instance(
        keyed_vectors=kv,
        data_set=DataSet.WN18,
        is_reflexive_match_allowed=False,
    )
    assert function_instance is not None

    h = "09590495"
    t = "09689152"

    result_h_prediction = function_instance.predict_heads([h, 'od:F3_had_source / ecrm:P137_exemplifies', t], n=None)
    # make sure that the tail is not predicted when predicting heads
    assert t not in result_h_prediction

    # make sure that the correct h appears in the prediction
    assert h in [x[0] for x in result_h_prediction]

    # make sure that the returned list is sorted in descending order of the confidence
    smallest_confidence = 100.0
    for p, confidence in result_h_prediction:
        assert (
                smallest_confidence >= confidence
        ), f"Result not correctly sorted for {function}"
        if confidence < smallest_confidence:
            smallest_confidence = confidence

    # perform test on tail prediction
    result_t_prediction = function_instance.predict_tails([h, l, t], n=None)

    # make sure that the head is not predicted when predicting tails
    assert (
            h not in result_t_prediction
    ), f"Failure for {function}: Found head as prediction of tail."

    # make sure that the solution appears for tail predictions
    assert t in [
        x[0] for x in result_t_prediction
    ], f"Failure for {function}: Did not find correct prediction of tail."

    smallest_confidence = 100.0
    for p, confidence in result_t_prediction:
        assert (
                smallest_confidence >= confidence
        ), f"Result not correctly sorted for {function}"
        if confidence < smallest_confidence:
            smallest_confidence = confidence

    # allow reflexive (but exclude most similar due to implementation of gensim excluding those always)
    if (
            function == PredictionFunctionEnum.MOST_SIMILAR
            or function == PredictionFunctionEnum.PREDICATE_AVERAGING_MOST_SIMILAR
    ):
        continue

    function_instance = function.get_instance(
        keyed_vectors=kv, data_set=DataSet.WN18, is_reflexive_match_allowed=True
    )
    assert function_instance is not None

    # make sure the solution is found (head prediction)
    result = function_instance.predict_heads([h, l, t], n=None)
    assert h in (
        item[0] for item in result
    ), f"Head {h} not found in prediction of function {function}."

    # make sure the solution is found (tail prediction)
    result = function_instance.predict_tails([h, l, t], n=None)
    assert t in (
        item[0] for item in result
    ), f"Tail {t} not found in prediction of function {function}."
