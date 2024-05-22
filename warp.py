import math
import torch


'''
This function converts the head pose predictions to degrees.
It takes the predicted head pose tensor (pred) as input.
It creates an index tensor (idx_tensor) with the same length as the head pose tensor.
It performs a weighted sum of the head pose predictions multiplied by the index tensor.
The result is then scaled and shifted to obtain the head pose in degrees.
'''
def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx, _ in enumerate(pred)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = pred.squeeze()
    pred = torch.sum(pred * idx_tensor) * 3 - 99
    return pred


'''
This function computes the rotation matrix based on the yaw, pitch, and roll angles.
It takes the yaw, pitch, and roll angles (in degrees) as input.
It converts the angles from degrees to radians using torch.deg2rad.
It creates separate rotation matrices for roll, pitch, and yaw using the corresponding angles.
It combines the rotation matrices using Einstein summation (torch.einsum) to obtain the final rotation matrix.
'''
def get_rotation_matrix(yaw, pitch, roll):
    yaw = torch.deg2rad(yaw)
    pitch = torch.deg2rad(pitch)
    roll = torch.deg2rad(roll)

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.zeros(roll.shape[0], 3, 3).to(roll.device)
    roll_mat[:, 0, 0] = torch.cos(roll)
    roll_mat[:, 0, 1] = -torch.sin(roll)
    roll_mat[:, 1, 0] = torch.sin(roll)
    roll_mat[:, 1, 1] = torch.cos(roll)
    roll_mat[:, 2, 2] = 1

    pitch_mat = torch.zeros(pitch.shape[0], 3, 3).to(pitch.device)
    pitch_mat[:, 0, 0] = torch.cos(pitch)
    pitch_mat[:, 0, 2] = torch.sin(pitch)
    pitch_mat[:, 1, 1] = 1
    pitch_mat[:, 2, 0] = -torch.sin(pitch)
    pitch_mat[:, 2, 2] = torch.cos(pitch)

    yaw_mat = torch.zeros(yaw.shape[0], 3, 3).to(yaw.device)
    yaw_mat[:, 0, 0] = torch.cos(yaw)
    yaw_mat[:, 0, 2] = -torch.sin(yaw)
    yaw_mat[:, 1, 1] = 1
    yaw_mat[:, 2, 0] = torch.sin(yaw)
    yaw_mat[:, 2, 2] = torch.cos(yaw)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', yaw_mat, pitch_mat, roll_mat)
    return rot_mat



'''
This function creates a coordinate grid based on the given spatial size.
It takes the spatial size (spatial_size) and data type (type) as input.
It creates 1D tensors (x, y, z) representing the coordinates along each dimension.
It normalizes the coordinate values to the range [-1, 1].
It meshes the coordinate tensors using broadcasting to create a 3D coordinate grid.
The resulting coordinate grid has shape (height, width, depth, 3), where the last dimension represents the (x, y, z) coordinates.
'''
def make_coordinate_grid(spatial_size, type):
    d, h, w = spatial_size
    x = torch.arange(w).to(type)
    y = torch.arange(h).to(type)
    z = torch.arange(d).to(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)

    yy = y.view(-1, 1, 1).repeat(1, w, d)
    xx = x.view(1, -1, 1).repeat(h, 1, d)
    zz = z.view(1, 1, -1).repeat(h, w, 1)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)
    return meshed

def compute_rt_warp2(rt, v_s, inverse=False):
    bs, _, d, h, w = v_s.shape
    yaw, pitch, roll = rt['yaw'], rt['pitch'], rt['roll']
    yaw = headpose_pred_to_degree(yaw)
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    # Invert the transformation matrix if needed
    if inverse:
        rot_mat = torch.inverse(rot_mat)

    rot_mat = rot_mat.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
    rot_mat = rot_mat.repeat(1, d, h, w, 1, 1)

    identity_grid = make_coordinate_grid((d, h, w), type=v_s.type())
    identity_grid = identity_grid.view(1, d, h, w, 3)
    identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

    t = t.view(t.shape[0], 1, 1, 1, 3)

    # Rotate
    warp_field = torch.bmm(identity_grid.reshape(-1, 1, 3), rot_mat.reshape(-1, 3, 3))
    warp_field = warp_field.reshape(identity_grid.shape)
    warp_field = warp_field - t

    return warp_field