import numpy as np 
import torch
#https://github.com/Yuol96/pytorch-triplet-loss/blob/master/model/tests/test_triplet_loss.py

def pairwise_distances_pytorch(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.mm(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diagonal(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = distances.clamp(min=0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        epsilon=1e-16
        mask = torch.eq(distances, 0).float()
        distances = distances + mask * epsilon

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    return distances

def get_valid_triplets_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    i_eq_j = label_equal.unsqueeze(2)
    i_eq_k = label_equal.unsqueeze(1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k

    mask = distinct_indices & valid_labels
    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = distances.unsqueeze(2)
    anchor_negative_dist = distances.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = get_triplet_mask(labels)
    triplet_loss = triplet_loss * mask.float()

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = triplet_loss.clamp(min=0)

    # Count number of positive triplets (where triplet_loss > 0)
    epsilon = 1e-16
    num_positive_triplets = (triplet_loss > 0).float().sum()
    num_valid_triplets = mask.float().sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

    return triplet_loss, fraction_positive_triplets

def get_valid_positive_mask(labels):
    """
    To be a valid positive pair (a,p),
        - a and p are different embeddings
        - a and p have the same label
    """
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_equal
    return mask


def get_valid_negative_mask(labels):
    """
    To be a valid negative pair (a,n),
        - a and n are different embeddings
        - a and n have the different label
    """
    indices_equal = torch.eye(labels.size(0)).bool()
    indices_not_equal = ~indices_equal

    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_not_equal
    return mask


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = anchor_positive_dist.max(dim=1)[0] 

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = pairwise_dist.max(dim=1,keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = anchor_negative_dist.min(dim=1)[0]

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = (hardest_positive_dist - hardest_negative_dist + margin).clamp(min=0)

    # Get final mean triplet loss
    triplet_loss = triplet_loss.mean()

    return triplet_loss


def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
            pairwise_distances.diagonal())
    return pairwise_distances

def test_pairwise_distances():
    """Test the pairwise distances function."""
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]  # to get distance 0

    for squared in [True, False]:
    	res_np = pairwise_distance_np(embeddings, squared=squared)
    	res_pytorch = pairwise_distances_pytorch(torch.from_numpy(embeddings), squared=squared)
    	assert np.allclose(res_np, res_pytorch.numpy())

def test_anchor_positive_triplet_mask():
    """Test function _get_anchor_positive_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            distinct = (i != j)
            valid = labels[i] == labels[j]
            mask_np[i, j] = (distinct and valid)

    mask_pytorch = get_valid_positive_mask(torch.from_numpy(labels))

    assert np.allclose(mask_np, mask_pytorch.numpy())

def test_anchor_negative_triplet_mask():
    """Test function _get_anchor_negative_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for k in range(num_data):
            distinct = (i != k)
            valid = (labels[i] != labels[k])
            mask_np[i, k] = (distinct and valid)

    mask_pytorch = get_valid_negative_mask(torch.from_numpy(labels))

    assert np.allclose(mask_np, mask_pytorch.numpy())

def test_triplet_mask():
    """Test function _get_triplet_mask."""
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    mask_np = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                mask_np[i, j, k] = (distinct and valid)

    mask_pytorch = get_valid_triplets_mask(torch.from_numpy(labels)).numpy()

    assert np.allclose(mask_np, mask_pytorch)

def test_batch_all_triplet_loss():
    """Test the triplet loss with batch all triplet mining"""
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    for squared in [True, False]:
        pdist_matrix = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        num_positives = 0.0
        num_valid = 0.0
        for i in range(num_data):
            for j in range(num_data):
                for k in range(num_data):
                    distinct = (i != j and i != k and j != k)
                    valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                    if distinct and valid:
                        num_valid += 1.0

                        pos_distance = pdist_matrix[i][j]
                        neg_distance = pdist_matrix[i][k]

                        loss = np.maximum(0.0, pos_distance - neg_distance + margin)
                        loss_np += loss

                        num_positives += (loss > 0)

        loss_np /= num_positives

        # Compute the loss in TF.
        loss_pytorch, fraction = batch_all_triplet_loss(torch.from_numpy(labels), torch.from_numpy(embeddings), margin, squared=squared)

        assert np.allclose(loss_np, loss_pytorch.item())
        assert np.allclose(num_positives / num_valid, fraction.item())

def test_batch_hard_triplet_loss():
    """Test the triplet loss with batch hard triplet mining"""
    num_data = 50
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data)).astype(np.float32)

    for squared in [True, False]:
        pdist_matrix = pairwise_distance_np(embeddings, squared=squared)

        loss_np = 0.0
        for i in range(num_data):
            # Select the hardest positive
            max_pos_dist = np.max(pdist_matrix[i][labels == labels[i]])

            # Select the hardest negative
            min_neg_dist = np.min(pdist_matrix[i][labels != labels[i]])

            loss = np.maximum(0.0, max_pos_dist - min_neg_dist + margin)
            loss_np += loss

        loss_np /= num_data

        # Compute the loss in TF.
        loss_pytorch = batch_hard_triplet_loss(torch.from_numpy(labels), torch.from_numpy(embeddings), margin, squared=squared)
        assert np.allclose(loss_np, loss_pytorch.item())