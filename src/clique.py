"""A collection of methods for clique finding in the explanations."""
from typing import Collection

from tqdm import tqdm

import numpy as np
from shapiq import InteractionValues, powerset


def get_interesting_starting_players(
        attribution_values: np.ndarray, 
        first_order_values: np.ndarray,
        k: int
    ) -> set[int]:
    players = set()
    sorted_attributions = np.argsort(attribution_values)
    sorted_first_order = np.argsort(first_order_values)
    players.update(sorted_attributions[-k:])
    players.update(sorted_first_order[-k:])
    players.update(sorted_attributions[:k])
    players.update(sorted_first_order[:k])
    # add values close to 0 to inrease exploration
    abs_sorted_attributions = np.argsort(np.abs(attribution_values))
    abs_sorted_first_order = np.argsort(np.abs(first_order_values))
    players.update(abs_sorted_attributions[:k])
    players.update(abs_sorted_first_order[:k])
    return players


def incremental_gain(iv: InteractionValues, v: int, S: Collection[int]) -> float:
    """Computes the incremental gain of adding a player to the clique.

    Args:
        iv: The interaction values.
        v: The player to add.
        S: The current clique.

    Returns:
        The incremental gain of adding the player to the clique.
    """
    gain = iv[(v,)]
    for u in S:
        gain += iv[(u, v)]
    return gain


def get_cliques_brute_force(
    iv: InteractionValues,
    max_size: int | None = None,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Uses a brute force algorithm to find a clique of size k with the largest or smallest value.

    Args:
        iv: The interaction values.
        max_size: The maximum size of the cliques.
        reverse: Whether to reverse mif/lif so that to insert instead of delete.

    Returns:
        The cliques_mif and cliques_lif matrices.
    """
    max_size = max_size if max_size is not None else iv.n_players
    cliques_mif = np.zeros((max_size, iv.n_players), dtype=int)
    cliques_lif = np.zeros((max_size, iv.n_players), dtype=int)
    all_cliques = {}
    for subset in powerset(range(iv.n_players), min_size=1, max_size=max_size):
        value = 0
        for interaction in powerset(subset, min_size=1):
            value += iv[interaction]
        all_cliques[subset] = value
    for size in range(1, max_size + 1):
        cliques_size = {k: v for k, v in all_cliques.items() if len(k) == size}
        clique_mif = max(cliques_size, key=cliques_size.get)
        clique_lif = min(cliques_size, key=cliques_size.get)
        cliques_mif[size - 1, list(clique_mif)] = 1
        cliques_lif[size - 1, list(clique_lif)] = 1
    if reverse:
        return cliques_lif[::-1], cliques_mif[::-1]
    else:
        return cliques_mif, cliques_lif


def get_cliques_greedy_mif_lif(
    iv: InteractionValues,
    start_players: set[int] | None = None,
    max_size: int | None = None,
    reverse: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Uses a greedy algorithm to find a clique of size k with the largest or smallest value.

    Args:
        iv: The interaction values.
        start_players: The players to start the search from.
        max_size: The maximum size of the cliques.
        reverse: Whether to reverse mif/lif so that to insert instead of delete.
        verbose: Whether to print progress information.

    Returns:
        The cliques_mif and cliques_lif matrices.
    """
    players = set(range(iv.n_players))
    if start_players is None:
        start_players = set(range(iv.n_players))
    pbar = tqdm(total=len(start_players), desc="Computing Cliques") if verbose else None
    cliques = {}
    max_size = max_size if max_size is not None else iv.n_players
    for player in start_players:
        clique_mif, clique_lif = {player}, {player}
        clique_value_mif, clique_value_lif = iv[(player,)], iv[(player,)]
        remaining_players_mif = players - clique_mif
        remaining_players_lif = players - clique_lif
        cliques[tuple(sorted(clique_mif))] = clique_value_mif
        for _ in range(max_size - 1):
            next_player_mif = max(remaining_players_mif, key=lambda p: incremental_gain(iv, p, clique_mif))
            next_player_lif = min(remaining_players_lif, key=lambda p: incremental_gain(iv, p, clique_lif))
            gain_mif = incremental_gain(iv, next_player_mif, clique_mif)
            gain_lif = incremental_gain(iv, next_player_lif, clique_lif)
            clique_mif.add(next_player_mif)
            clique_lif.add(next_player_lif)
            clique_value_mif += gain_mif
            clique_value_lif += gain_lif
            remaining_players_mif.remove(next_player_mif)
            remaining_players_lif.remove(next_player_lif)
            cliques[tuple(sorted(clique_mif))] = clique_value_mif
            cliques[tuple(sorted(clique_lif))] = clique_value_lif
        if pbar is not None:
            pbar.update(1)
    cliques_mif = np.zeros((iv.n_players, iv.n_players), dtype=int)
    cliques_lif = np.zeros((iv.n_players, iv.n_players), dtype=int)
    cliques_per_size = {}
    for k, v in cliques.items():
        size = len(k)
        if size in cliques_per_size:
            cliques_per_size[size][k] = v
        else:
            cliques_per_size[size] = {k: v}
    for size in range(1, max_size + 1):
        cliques_size = cliques_per_size[size]
        clique_mif = max(cliques_size, key=cliques_size.get)
        clique_lif = min(cliques_size, key=cliques_size.get)
        cliques_mif[size - 1, list(clique_mif)] = 1
        cliques_lif[size - 1, list(clique_lif)] = 1
    if reverse:
        return cliques_lif[::-1], cliques_mif[::-1]
    else:
        return cliques_mif, cliques_lif


def get_incremental_gain_memoizer(iv: InteractionValues):
    # Cache for singleton and pairwise interactions
    singleton_cache = {}
    pairwise_cache = {}

    def incremental_gain(v: int, S: Collection[int]) -> float:
        # Get singleton term
        if v not in singleton_cache:
            singleton_cache[v] = iv[(v,)]
        gain = singleton_cache[v]

        for u in S:
            key = tuple(sorted((u, v)))  # symmetric key for (u, v)
            if key not in pairwise_cache:
                pairwise_cache[key] = iv[key]
            gain += pairwise_cache[key]

        return gain

    return incremental_gain


def get_clique_value(iv: InteractionValues, clique: set[int]) -> float:
    """Computes the value of a clique for a given interaction value."""
    score = 0
    for interaction in iv.interaction_lookup.keys():
        if len(interaction) == 0:
            continue
        if all(i in clique for i in interaction):
            score += iv[interaction]
    return score
