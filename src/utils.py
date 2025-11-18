import random
import warnings
from PIL import Image

import torch
import shapiq
import numpy as np
from shapiq import InteractionValues


def save_game(path, game, sampling_adjustment_weights):
    if not path.endswith(".npz"):
        path += ".npz"
    game.value_storage.astype(np.float16)
    coalitions_in_storage = shapiq.utils.transform_coalitions_to_array(
        coalitions=game.coalition_lookup, n_players=game.n_players
    ).astype(bool)
    np.savez_compressed(
        path,
        values=game.value_storage,
        coalitions=coalitions_in_storage,
        sampling_adjustment_weights=sampling_adjustment_weights,
        n_players=game.n_players,
        normalization_value=game.normalization_value,
    )


def convert_iv_to_first_order(iv: shapiq.InteractionValues, p_sampler=0.5):
    """p_sampler for Banzhaf and Shapley is 0.5"""
    if iv.max_order == 1:
        warnings.warn("Input to convert_iv_to_first_order() has max_order=1. Returning the object.")
        return iv
    dict_sv = {(): iv.dict_values[()]}
    dict_sv = {k: v for k, v in iv.dict_values.items() if len(k) == 1}
    for i in range(iv.n_players):
        for j in range(i + 1, iv.n_players):
            if (i, j) in iv.dict_values:
                contr = p_sampler * iv.dict_values[(i, j)]
                dict_sv[(i,)] += contr
                dict_sv[(j,)] += contr

    return shapiq.InteractionValues(
        values=np.array([v for _, v in dict_sv.items()]),
        index=iv.index,
        max_order=1,
        n_players=iv.n_players,
        min_order=iv.min_order,
        interaction_lookup={k: i for i, (k, _) in enumerate(dict_sv.items())},
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )


def convert_iv_to_first_order(iv: shapiq.InteractionValues, p_sampler=0.5):
    """p_sampler for Banzhaf and Shapley is 0.5"""
    if iv.max_order == 1:
        warnings.warn("Input to convert_iv_to_first_order() has max_order=1. Returning the object.")
        return iv
    dict_sv = {(): iv.dict_values[()]}
    dict_sv = {k: v for k, v in iv.dict_values.items() if len(k) == 1}
    for i in range(iv.n_players):
        for j in range(i + 1, iv.n_players):
            if (i, j) in iv.dict_values:
                contr = p_sampler * iv.dict_values[(i, j)]
                dict_sv[(i,)] += contr
                dict_sv[(j,)] += contr

    return shapiq.InteractionValues(
        values=np.array([v for _, v in dict_sv.items()]),
        index=iv.index,
        max_order=1,
        n_players=iv.n_players,
        min_order=iv.min_order,
        interaction_lookup={k: i for i, (k, _) in enumerate(dict_sv.items())},
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )

def convert_exclip_to_first_order(iv: shapiq.InteractionValues, n_players_image, n_players_text):
    dict_sv = {(): iv.dict_values[()]}
    dict_sv = {k: v for k, v in iv.dict_values.items() if len(k) == 1}
    for i in range(n_players_image):
        for j in range(n_players_image, n_players_image + n_players_text):
            if (i, j) in iv.dict_values:
                contr = iv.dict_values[(i, j)]
                dict_sv[(i,)] += contr / n_players_text
                dict_sv[(j,)] += contr / n_players_image

    return shapiq.InteractionValues(
        values=np.array([v for _, v in dict_sv.items()]),
        index=iv.index,
        max_order=1,
        n_players=iv.n_players,
        min_order=iv.min_order,
        interaction_lookup={k: i for i, (k, _) in enumerate(dict_sv.items())},
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )


def convert_array_to_first_order(array, index=""):
    if len(array.shape) == 2:
        array = array.reshape(-1)
    values = {(): 0}
    values = {(k,): v for k, v in enumerate(array)}

    return shapiq.InteractionValues(
        values=np.array([v for _, v in values.items()]),
        index=index,
        max_order=1,
        n_players=len(values),
        min_order=0,
        interaction_lookup={k: i for i, (k, _) in enumerate(values.items())},
        baseline_value=0,
    )


def convert_array_to_second_order(array, index=""):
    assert len(array.shape) == 2
    n_players_image = array.shape[1]
    n_players_text = array.shape[0]
    values = [0]
    interaction_lookup = {(): 0}
    for t in range(n_players_text):
        values.append(0)
        interaction_lookup[(t,)] = len(interaction_lookup)   
    for i in range(n_players_text, n_players_text + n_players_image):
        values.append(0)
        interaction_lookup[(i,)] = len(interaction_lookup)   
    for t in range(n_players_text):
        for i in range(n_players_image):
            values.append(array[t, i].item())
            interaction_lookup[(i, n_players_image + t)] = len(interaction_lookup)

    return shapiq.InteractionValues(
        values=np.array(values),
        index=index,
        max_order=2,
        n_players=n_players_text + n_players_image,
        min_order=0,
        interaction_lookup=interaction_lookup,
        baseline_value=0,
    )


def get_superset(iv: shapiq.InteractionValues, players: list[int], max_order: int | None = None):
    keys_in_subset, players_new, _max_order, _min_order = set(), set(), 0, 1e10
    for key in iv.interaction_lookup.keys():
        if any(p in players for p in key):
            if max_order is not None and len(key) > max_order:
                continue
            keys_in_subset.add(key)
            players_new.update({p for p in key})
            _max_order = max(_max_order, len(key))
            _min_order = min(_min_order, len(key))
    new_values = np.zeros(len(keys_in_subset))
    new_interaction_lookup = {}
    for index, key in enumerate(keys_in_subset):
        new_interaction_lookup[key] = index
        new_values[index] = iv[key]

    n_players = len(players_new)
    return shapiq.InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=_max_order,
        n_players=n_players,
        min_order=_min_order,
        interaction_lookup=new_interaction_lookup,
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )


def get_subset(iv: shapiq.InteractionValues, players: list[int], rename_players: bool = True):
    keys = iv.interaction_lookup.keys()
    idx, keys_in_subset = [], []
    for i, key in enumerate(keys):
        if all(p in players for p in key):
            idx.append(i)
            keys_in_subset.append(key)
    new_values = iv.values[idx]
    new_interaction_lookup = {}
    for index, key in enumerate(keys_in_subset):
        new_interaction_lookup[key] = index

    n_players = len(players)

    if rename_players:
        rename_dict = {p: i for i, p in enumerate(players)}
        renamed_lookup = {}
        for old_interaction, index in new_interaction_lookup.items():
            new_interaction = {rename_dict[p] for p in old_interaction}
            renamed_lookup[tuple(new_interaction)] = index
        new_interaction_lookup = renamed_lookup

    return shapiq.InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=iv.max_order,
        n_players=n_players,
        min_order=iv.min_order,
        interaction_lookup=new_interaction_lookup,
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )


def append_images(images, direction='horizontal',
                  bg_color=(255, 255, 255), aligment='center'):
    """ https://stackoverflow.com/a/46623632
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def get_max_min_values(ivs: list[shapiq.InteractionValues]) -> tuple[float, float]:
    """Get the maximum absolute value of the InteractionValues in the list of InteractionValues."""
    minimum = float("inf")
    maximum = float("-inf")
    for iv in ivs:
        max_order = iv.max_order
        iv_without_baseline = get_n_order(iv=iv, order=max_order, min_order=1)
        assert tuple() not in iv_without_baseline.interaction_lookup
        max_value = float(np.max(np.abs(iv_without_baseline.values)))
        maximum = max(maximum, max_value)
        min_value = float(np.min(np.abs(iv_without_baseline.values)))
        minimum = min(minimum, min_value)
    return maximum, minimum


def sort_interactions(
    iv: InteractionValues,
    reverse: bool = True,
    sort_by_abs: bool = True
) -> list[tuple[tuple[int, ...], float, float]]:
    """Sort the InteractionValues from highest to lowest.

    Args:
        iv: The InteractionValues to sort.
        reverse: Whether to sort in descending order. Defaults to ``True``.
        sort_by_abs: Whether to sort by absolute value. Defaults to ``True``. If ``False``, then
            sort by the value itself (positive values first).

    Returns:
        A list of tuples containing the interaction, the value, and the absolute value of the
            interaction.
    """
    sorted_interactions = []
    sort_item = 2 if sort_by_abs else 1
    for interaction in iv.interaction_lookup.keys():
        score = iv[interaction]
        item = (interaction, score, abs(score))
        sorted_interactions.append(item)
    sorted_interactions = sorted(sorted_interactions, key=lambda x: x[sort_item], reverse=reverse)
    return sorted_interactions


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_crossmodal_interaction_lookup(n_players_image, n_players_text):
    """Adds all first order interactions and only the interactions between n_players_image and n_players_text."""
    interaction_lookup = {(): 0}
    for i in range(n_players_image + n_players_text):
        interaction_lookup[(i,)] = len(interaction_lookup)
    for i in range(n_players_image):
        for j in range(n_players_image, n_players_image + n_players_text):
            interaction_lookup[(i, j)] = len(interaction_lookup)
    return interaction_lookup


def create_subset_interaction_lookup(n_players, players_clique):
    """Adds all first order interactions and only the subset of second order interactions."""
    interaction_lookup = {(): 0}
    for i in range(n_players):
        interaction_lookup[(i,)] = len(interaction_lookup)
    for i in players_clique:
        for j in players_clique:
            if i < j:
                interaction_lookup[(i, j)] = len(interaction_lookup)
    return interaction_lookup


def get_top_clique_players(attribution_values, size_clique, n_players_image, n_players_text):
    """Interactive approach."""
    av_image = get_subset(attribution_values, list(range(n_players_image)))
    av_text = get_subset(attribution_values, list(range(n_players_image, n_players_image + n_players_text)))
    size_clique_text = int(min(n_players_text, max(5, np.ceil(n_players_text * (size_clique / (n_players_image + n_players_text))))))
    size_clique_image = size_clique - size_clique_text
    players_clique = np.concatenate([
        np.argpartition(np.abs(av_image.get_n_order(1).values), -size_clique_image)[-size_clique_image:],
        np.argpartition(np.abs(av_text.get_n_order(1).values), -size_clique_text)[-size_clique_text:] + n_players_image,
    ])
    assert len(players_clique) == size_clique
    return players_clique


def get_crossmodal_subset(iv, n_players_image, n_players_text):
    crossmodal_interaction_lookup = create_crossmodal_interaction_lookup(n_players_image, n_players_text)
    idx, keys_in_subset = [], []
    for key, value in iv.interaction_lookup.items():
        if key in crossmodal_interaction_lookup:
            idx.append(value)
            keys_in_subset.append(key)
    new_values = iv.values[idx]
    new_interaction_lookup = {}
    for index, key in enumerate(keys_in_subset):
        new_interaction_lookup[key] = index

    return shapiq.InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=iv.max_order,
        n_players=n_players_image+n_players_text,
        min_order=iv.min_order,
        interaction_lookup=new_interaction_lookup,
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )


def get_n_order(
    iv: InteractionValues,
    order: int,
    min_order: int | None = None,
    max_order: int | None = None
) -> "InteractionValues":
    """Returns the interaction values of a specific order.

    Args:
        iv: The InteractionValues to get the specific order from.
        order: The order of the interactions to return.
        min_order: The minimum order of the interactions to return. Defaults to ``None`` which
            sets it to the order.
        max_order: The maximum order of the interactions to return. Defaults to ``None`` which
            sets it to the order.

    Returns:
        The interaction values of the specified order.
    """
    max_order = order if max_order is None else max_order
    min_order = order if min_order is None else min_order

    new_values = []
    new_interaction_lookup = {}
    for interaction in iv.interaction_lookup.keys():
        if len(interaction) > max_order or len(interaction) < min_order:
            continue
        new_interaction_lookup[interaction] = len(new_interaction_lookup)
        new_values.append(iv[interaction])

    return InteractionValues(
        values=np.array(new_values, dtype=float),
        index=iv.index,
        max_order=order,
        n_players=iv.n_players,
        min_order=order,
        interaction_lookup=new_interaction_lookup,
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )


def denormalize(img, mean, std):
    return img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)


def get_conditioned_interactions(iv: InteractionValues, player: int, divide: bool = False) -> InteractionValues:
    """Creates a new InteractionValues object with only the interactions that include the
    specified distributed to the first order."""
    if iv.max_order == 1:
        warnings.warn("Input to get_conditioned_interactions() has max_order=1. Returning the object.")
        return iv
    if iv.max_order > 2:
        raise NotImplementedError(f"max_order={iv.max_order} not implemented. This code only makes sense for max_order=2.")
    keys_in_subset = set()
    for interaction in iv.interaction_lookup.keys():
        if player in interaction and len(interaction) > 1:
            keys_in_subset.add(interaction)
    new_values = np.zeros(len(keys_in_subset))
    new_interaction_lookup = {}
    for index, interaction in enumerate(keys_in_subset):
        if interaction[0] != player:
            other_player = interaction[0]
        else:
            other_player = interaction[1]
        score = iv[interaction]
        if divide:
            score = score / (len(interaction))
        new_interaction_lookup[(other_player,)] = index
        new_values[index] = score

    # add original player with epsilon value
    new_interaction_lookup[(player,)] = len(new_interaction_lookup)
    epsilon_score = 1e-10
    new_values = np.concatenate([new_values, [epsilon_score]])

    # add all interactions that are missing in the new interaction lookup with an epsilon value
    additional_values = []
    for i in range(iv.n_players):
        if (i,) not in new_interaction_lookup:
            new_interaction_lookup[(i,)] = len(new_interaction_lookup)
            additional_values.append(epsilon_score)
    additional_values = np.array(additional_values)
    new_values = np.concatenate([new_values, additional_values])

    return shapiq.InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=1,
        n_players=iv.n_players,
        min_order=1,
        interaction_lookup=new_interaction_lookup,
        estimated=iv.estimated,
        estimation_budget=iv.estimation_budget,
        baseline_value=iv.baseline_value,
    )