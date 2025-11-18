from typing import Literal

import numpy as np
from shapiq import InteractionValues
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
import scipy as sp

from src.game_huggingface import VisionLanguageGame
from src.sampler import CoalitionSampler


def get_explanation_output(
    iv: InteractionValues,
    coalition_matrix: np.ndarray,
) -> np.ndarray:
    """ Evaluate the explanation on the sampled coalitions.

    Args:
        iv: The interaction values.
        coalition_matrix: The sampled coalitions.

    Returns:
        The explanation output.
    """
    outputs = np.zeros(coalition_matrix.shape[0])
    for i in range(coalition_matrix.shape[0]):
        coalition = coalition_matrix[i]
        # turn one-hot vector into a tuple
        coalition = tuple(np.where(coalition)[0])
        for interaction in iv.interaction_lookup.keys():
            # check if the interaction is a subset of the coalition
            if all(player in coalition for player in interaction):
                outputs[i] += iv[interaction]
    return outputs


def compute_weighted_r2(
    gt_values: np.ndarray,
    et_values: np.ndarray,
    coalition_matrix: np.ndarray,
    weights: np.ndarray,
):
    """Compute a weighted R^2 score, where the weights are different for different coalition sizes.

    Args:
        gt_values: The ground truth values as a numpy array of shape (n_samples,).
        et_values: The explanation values as a numpy array of shape (n_samples,).
        coalition_matrix: The coalition matrix as a numpy array of shape (n_samples, n_players).
        weights: The weights for each coalition size as a numpy array of shape (n_sizes,).
    """
    n_players = np.shape(weights)[0] - 1
    weights_per_coal = np.zeros(gt_values.shape)
    for i in range(gt_values.shape[0]):
        coalition_size = np.sum(coalition_matrix[i])
        weights_per_coal[i] = weights[coalition_size]/sp.special.binom(n_players,coalition_size)

    # Compute the weighted mean of the ground truth values
    weighted_mean_gt = np.average(gt_values, weights=weights_per_coal)

    # Compute the weighted total sum of squares
    ss_total = np.sum(weights_per_coal * (gt_values - weighted_mean_gt) ** 2)

    # Compute the weighted residual sum of squares
    ss_residual = np.sum(weights_per_coal * (gt_values - et_values) ** 2)

    # Compute the weighted R^2 score
    r2 = 1 - (ss_residual / ss_total)
    return r2


def eval_faithfulness_one_game(
    game: VisionLanguageGame,
    explanations: dict[str, InteractionValues],
    n_eval_coalitions: int = 100,
    sample_mode: Literal["banzhaf"] | Literal["shapley"] = "banzhaf",
    sample_p: float = 0.5,
    instance_id: int | None = None
) -> list[dict[str, float]]:
    """Compute a faithfulness score for a set of explanations.

    The faithfulness scores are computed as follows:
        1. Sample a subset of $n$ coalitions.
        2. Evaluate the game on the sampled coalitions to create ground-truth values.
        3. For each explanation create an explanation output by summing all interactions which are
            a subset of a coalition and returning the sum.
        4. Compute Faithfulness scores between the ground-truth values and the explanation values.

    Args:
        game: The game object which can be used to query the model with removal.
        explanations: A dictionary mapping from explanation names to explanation values.
        n_eval_coalitions: The number of coalitions to evaluate.
        sample_mode: The sampling mode to use. Can be either "banzhaf" or "shapley".
        sample_p: The p to shift the distribution with (0.5 is centered).
        instance_id: An optional instance id to use for reproducibility.
            If None, a random seed is used.

    Returns:
        A list of faithfulness scores for different explanations.
    """

    # get sampling weights
    sampling_size_weights_banzhaf = np.array([
        sp.special.binom(game.n_players, k) * (sample_p ** k) * (
        (1 - sample_p) ** (game.n_players - k)) for k in range(game.n_players + 1)
    ])
    sampling_size_weights_shapley = np.zeros(game.n_players + 1)
    for coalition_size in range(1, game.n_players):
        sampling_size_weights_shapley[coalition_size] = 1 / (coalition_size * (game.n_players - coalition_size))

    # Sample coalitions
    if sample_mode == "banzhaf":
        sampling_size_weights = sampling_size_weights_banzhaf
        enforce_empty_full = False
    elif sample_mode == "shapley": # KernelSHAP sampling weights
        sampling_size_weights = sampling_size_weights_shapley
        enforce_empty_full = True
    else:
        raise ValueError(f"Unknown sampling mode: {sample_mode}")

    sampler = CoalitionSampler(
        n_players=game.n_players,
        sampling_weights=sampling_size_weights,
        enforce_empty_full=enforce_empty_full,
        pairing_trick=False,
        random_state=instance_id
    )
    sampler.sample(n_eval_coalitions)
    coalition_matrix = sampler.coalitions_matrix

    # get ground-truth values
    ground_truth_values = game.value_function(coalition_matrix)
    empty_prediction = game.normalization_value
    ground_truth_values -= empty_prediction

    # get explanation values and compute faithfulness scores
    scores = []
    for method_name, explanation in explanations.items():
        # Get explanation output
        explanation_output = get_explanation_output(explanation, coalition_matrix)

        # Compute faithfulness scores
        r2 = r2_score(ground_truth_values, explanation_output)
        r2_shapley = compute_weighted_r2(
            ground_truth_values,
            explanation_output,
            coalition_matrix,
            sampling_size_weights_shapley
        )
        r2_banzhaf = compute_weighted_r2(
            ground_truth_values,
            explanation_output,
            coalition_matrix,
            sampling_size_weights_banzhaf
        )
        mse = np.mean((ground_truth_values - explanation_output) ** 2)
        mae = np.mean(np.abs(ground_truth_values - explanation_output))
        pearson_corr = np.corrcoef(ground_truth_values, explanation_output)[0, 1]
        spearman_corr = sp.stats.spearmanr(ground_truth_values, explanation_output)[0]
        kendall_tau = sp.stats.kendalltau(ground_truth_values, explanation_output)[0]
        cosine_sim = cosine_similarity(ground_truth_values.reshape(1, -1), explanation_output.reshape(1, -1))[0][0]

        scores.append({
            "method_name": method_name,
            "instance_id": instance_id,
            "sample_mode": sample_mode,
            "sample_p": sample_p if sample_mode == "banzhaf" else None,
            "budget": explanation.estimation_budget,
            "n_players": game.n_players,
            "n_eval_coalitions": n_eval_coalitions,
            "r2": float(r2),
            "r2_shapley": float(r2_shapley),
            "r2_banzhaf": float(r2_banzhaf),
            "mse": float(mse),
            "mae": float(mae),
            "correlation": float(pearson_corr),
            "spearman_correlation": float(spearman_corr),
            "kendall_tau": float(kendall_tau),
            "cosine_similarity": float(cosine_sim)
        })
    return scores
