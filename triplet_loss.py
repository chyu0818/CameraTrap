import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class Embedder(nn.Module):
    def __init__(self, embedding_net):
        super(Embedder, self).__init__()
        self.embedding_net = embedding_net
        #self.fc1 = nn.Linear(1000, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, 267)
    
    def forward(self, x):
        x = self.embedding_net(x)
        #x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        #x = F.relu(x)
        #x = self.fc3(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 267)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
        
class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletLossBatchAll(nn.Module):
    """
    Triplet loss (batch all)
    https://omoindrot.github.io/triplet-loss#a-better-implementation-with-online-triplet-mining
    From embeddings, select all the valid triplets, and average the loss on the hard and semi-hard triplets
    """

    def __init__(self, margin=1.0):
        super(TripletLossBatchAll, self).__init__()
        self.margin = margin

    def pairwise_distances(self, embeddings, squared=False):
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
            distances += mask * epsilon

            distances = torch.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances *= (1.0 - mask)
        return distances


    def get_valid_triplets_mask(self, labels):
        """
        To be valid, a triplet (a,p,n) has to satisfy:
            - a,p,n are distinct embeddings
            - a and p have the same label, while a and n have different label
        """
        indices_equal = torch.eye(labels.size(0)).byte().cuda()
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


    def batch_all_triplet_loss(self, labels, embeddings, squared=False):
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
        pairwise_dist = self.pairwise_distances(embeddings, squared=squared)

        anchor_positive_dist = distances.unsqueeze(2)
        anchor_negative_dist = distances.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = self.get_triplet_mask(labels)
        triplet_loss = triplet_loss * mask.float()

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss.clamp_(min=0)

        # Count number of positive triplets (where triplet_loss > 0)
        epsilon = 1e-16
        num_positive_triplets = (triplet_loss > 0).float().sum()
        num_valid_triplets = mask.float().sum()
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

        return triplet_loss, fraction_positive_triplets


    def forward(self, embeddings, labels):
        triplet_loss, fraction_positive_triplets = self.batch_all_triplet_loss(labels, embeddings)
        return triplet_loss


class TripletLossBatchHard(nn.Module):
    """
    Triplet loss (batch all)
    https://omoindrot.github.io/triplet-loss#a-better-implementation-with-online-triplet-mining
    From embeddings, select all the valid triplets, and average the loss on the hard and semi-hard triplets
    """

    def __init__(self, margin=1.0):
        super(TripletLossBatchHard, self).__init__()
        self.margin = margin

    def pairwise_distances(self, embeddings, squared=False):
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
            distances += mask * epsilon

            distances = torch.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances *= (1.0 - mask)
        return distances


    def get_anchor_positive_triplet_mask(self, labels):
        """
        To be a valid positive pair (a,p),
            - a and p are different embeddings
            - a and p have the same label
        """
        indices_equal = torch.eye(labels.size(0)).byte().cuda()
        indices_not_equal = ~indices_equal

        label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).cuda()

        mask = indices_not_equal & label_equal
        return mask


    def get_anchor_negative_triplet_mask(self, labels):
        """
        To be a valid negative pair (a,n),
            - a and n are different embeddings
            - a and n have the different label
        """
        indices_equal = torch.eye(labels.size(0)).byte().cuda()
        indices_not_equal = ~indices_equal

        label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0)).cuda()

        mask = indices_not_equal & label_not_equal
        return mask


    def batch_hard_triplet_loss(self, labels, embeddings, squared=False):
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
        pairwise_dist = self.pairwise_distances(embeddings, squared=squared)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self.get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = mask_anchor_positive.float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist = anchor_positive_dist.max(dim=1)[0] 

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self.get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = mask_anchor_negative.float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist = pairwise_dist.max(dim=1,keepdim=True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist = anchor_negative_dist.min(dim=1)[0]

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

        # Get final mean triplet loss
        triplet_loss = triplet_loss.mean()

        return triplet_loss



    def forward(self, embeddings, labels):
        triplet_loss = self.batch_hard_triplet_loss(labels, embeddings)
        return triplet_loss
