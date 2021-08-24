import torch


def hausdorff(point_cloud1: torch.Tensor, point_cloud2: torch.Tensor):
    """
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3).repeat((1, 1, 1, n_pts2))  # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2).repeat((1, 1, n_pts1, 1))  # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, 1))  # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, 2)
    hausdorff_dist, _ = torch.max(shortest_dist, 1)  # (B, )

    return torch.mean(hausdorff_dist)
