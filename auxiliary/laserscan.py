#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.autograd import Function
# import emd


class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.reset()

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32, count=-1)
    scan = scan.reshape((-1, 5))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)

  def set_points(self, points, remissions=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.float32)


class SemLaserScan(LaserScan):
  """Class that contains LaserScan with x,y,z,r,sem_label,sem_color_label,inst_label,inst_color_label"""
  EXTENSIONS_LABEL = ['.label']

  def __init__(self, nclasses, sem_color_dict=None, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    super(SemLaserScan, self).__init__(project, H, W, fov_up, fov_down)
    self.reset()
    self.nclasses = nclasses         # number of classes

    # make semantic colors
    max_sem_key = 0
    for key, data in sem_color_dict.items():
      if key + 1 > max_sem_key:
        max_sem_key = key + 1
    self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
    for key, value in sem_color_dict.items():
      self.sem_color_lut[key] = np.array(value, np.float32) / 255.0

    # make instance colors
    max_inst_id = 100000
    self.inst_color_lut = np.random.uniform(low=0.0,
                                            high=1.0,
                                            size=(max_inst_id, 3))
    # force zero to a gray-ish color
    self.inst_color_lut[0] = np.full((3), 0.1)

  def reset(self):
    """ Reset scan members. """
    super(SemLaserScan, self).reset()

    # semantic labels
    self.sem_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.sem_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # instance labels
    self.inst_label = np.zeros((0, 1), dtype=np.uint32)         # [m, 1]: label
    self.inst_label_color = np.zeros((0, 3), dtype=np.float32)  # [m ,3]: color

    # projection color with semantic labels
    self.proj_sem_label = np.zeros((self.proj_H, self.proj_W),
                                   dtype=np.int32)              # [H,W]  label
    self.proj_sem_color = np.zeros((self.proj_H, self.proj_W, 3),
                                   dtype=np.float)              # [H,W,3] color

    # projection color with instance labels
    self.proj_inst_label = np.zeros((self.proj_H, self.proj_W),
                                    dtype=np.int32)              # [H,W]  label
    self.proj_inst_color = np.zeros((self.proj_H, self.proj_W, 3),
                                    dtype=np.float)              # [H,W,3] color
    self.uncertainty_scores = None

  def open_label(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # if all goes well, open label
    label = np.load(filename)['data']
    label = label.reshape((-1))

    # set it
    self.set_label(label)

  def open_uncertainty(self, filename):
    """ Open raw scan and fill in attributes
    """
    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_LABEL):
      raise RuntimeError("Filename extension is not valid label file.")

    # if all goes well, open label
    uncertainty = np.fromfile(filename, dtype=np.float32)
    uncertainty = uncertainty.reshape((-1))

    self.uncertainty_scores = uncertainty
    uncertainty_min = np.min(self.uncertainty_scores)
    uncertainty_max = np.max(self.uncertainty_scores)
    self.uncertainty_scores = (self.uncertainty_scores * 255).astype(np.uint8)
    # self.uncertainty_scores = ((self.uncertainty_scores - uncertainty_min) / (uncertainty_max - uncertainty_min) * 255).astype(np.uint8)
    # for i in range(10):
    #   print(i/10, np.sum(self.uncertainty_scores<i/10) / self.uncertainty_scores.shape[0])


  def set_label(self, label):
    """ Set points for label not from file but from np
    """
    # check label makes sense
    if not isinstance(label, np.ndarray):
      raise TypeError("Label should be numpy array")

    # only fill in attribute if the right size
    if label.shape[0] == self.points.shape[0]:
      self.sem_label = (label // 1000).astype(np.uint8)  # semantic label in lower half
      self.inst_label = (label % 1000).astype(np.uint8)    # instance id in upper half
      cls, cnt = np.unique(self.sem_label, return_counts=True)
      unknown_clss = [9,12,18,22]
      for unknown_cls in unknown_clss:
        if unknown_cls in np.unique(self.sem_label):
          print(unknown_cls, cnt[cls==unknown_cls])
    else:
      print("Points shape: ", self.points.shape)
      print("Label shape: ", label.shape)
      raise ValueError("Scan and Label don't contain same number of points")

    # sanity check
    assert((self.inst_label + (self.sem_label * 1000) == label).all())

    # self.augmentor()

    if self.project:
      self.do_label_projection()

  def colorize(self):
    """ Colorize pointcloud with the color of each semantic label
    """
    self.sem_label_color = self.sem_color_lut[self.sem_label]
    self.sem_label_color = self.sem_label_color.reshape((-1, 3))

    self.inst_label_color = self.inst_color_lut[self.inst_label]
    self.inst_label_color = self.inst_label_color.reshape((-1, 3))

    # if self.uncertainty_scores is not None:
    #   self.uncertainty_color = np.zeros((3, self.uncertainty_scores.shape[0]))
    #   self.uncertainty_color[1] = 1 - self.uncertainty_scores
    #   self.uncertainty_color[2] = self.uncertainty_scores
    #   self.uncertainty_color = self.uncertainty_color.transpose(1,0)

    if self.uncertainty_scores is not None:
      viridis_map = self.get_mpl_colormap("viridis")
      self.uncertainty_color = viridis_map[self.uncertainty_scores]

  def do_label_projection(self):
    # only map colors to labels that exist
    mask = self.proj_idx >= 0

    # semantics
    self.proj_sem_label[mask] = self.sem_label[self.proj_idx[mask]]
    self.proj_sem_color[mask] = self.sem_color_lut[self.sem_label[self.proj_idx[mask]]]

    # instances
    self.proj_inst_label[mask] = self.inst_label[self.proj_idx[mask]]
    self.proj_inst_color[mask] = self.inst_color_lut[self.inst_label[self.proj_idx[mask]]]

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def augmentor(self):
    valid = self.sem_label != 20
    self.sem_label_valid = self.sem_label[valid]
    self.inst_label_valid = self.inst_label[valid]
    self.points_valid = self.points[valid]

    minimum_pts_thre = 100
    cls, cnt = np.unique(self.inst_label_valid, return_counts=True)
    inst_basic_idx = cls[cnt >= minimum_pts_thre][1:]

    for instance_idx in inst_basic_idx:
      obj_ins = self.points_valid[self.inst_label_valid == instance_idx]
      obj_ins_center = np.mean(obj_ins, axis=0)
      obj_ins = obj_ins - obj_ins_center
      scale_ds_large = np.random.rand() * 1.5 + 1.5
      scale_ds_small = np.random.rand() * 0.25 + 0.25
      rnd = np.random.rand()
      scale_ds = scale_ds_large if rnd > 0.5 else scale_ds_small
      obj_ins = obj_ins * scale_ds + obj_ins_center
      self.points_valid[self.inst_label_valid == instance_idx] = obj_ins
      self.sem_label_valid[self.inst_label_valid == instance_idx] = 0

    # inst_basic_num = cnt[cnt >= minimum_pts_thre][1:]
    # index_perm = np.random.permutation(inst_basic_idx.shape[0])
    # inst_basic_idx_perm = np.expand_dims(inst_basic_idx[index_perm], 0)
    # inst_basic_num_perm = np.expand_dims(inst_basic_num[index_perm], 0)
    # inst_basic_idx = np.expand_dims(inst_basic_idx, 0)
    # inst_basic_num = np.expand_dims(inst_basic_num, 0)
    # inst_num_pair = np.concatenate([inst_basic_num, inst_basic_num_perm], axis=0).transpose()
    # inst_idx_pair = np.concatenate([inst_basic_idx, inst_basic_idx_perm], axis=0).transpose()
    # min_pts_num_pair = np.min(inst_num_pair, axis=1)
    # print(inst_num_pair)
    #
    # obj_aug = []
    # idx_final = np.ones(self.points_valid.shape[0]).astype(np.bool_)
    #
    # for i in range(inst_idx_pair.shape[0]):
    #   obj_1 = self.points_valid[self.inst_label_valid == inst_idx_pair[i,0]]
    #   obj_2 = self.points_valid[self.inst_label_valid == inst_idx_pair[i,1]]
    #   min_pts = min_pts_num_pair[i] if min_pts_num_pair[i] < 1024 else 1024
    #   print(min_pts)
    #   np.random.shuffle(obj_1)
    #   obj_1 = obj_1[:min_pts]
    #   np.random.shuffle(obj_2)
    #   obj_2 = obj_2[:min_pts]
    #   center_obj_1 = np.mean(obj_1, axis=0)
    #   center_obj_2 = np.mean(obj_2, axis=0)
    #   obj_1 = obj_1 - center_obj_1
    #   obj_1_tmp = (obj_1 - np.min(obj_1, axis=0))/ (np.max(obj_1, axis=0) - np.min(obj_1, axis=0))
    #   obj_2 = obj_2 - center_obj_2
    #   obj_2_tmp = (obj_2 - np.min(obj_2, axis=0)) / (np.max(obj_2, axis=0) - np.min(obj_2, axis=0))
    #   if obj_1.shape[0] < 1024:
    #     obj_1_tmp = torch.from_numpy(np.concatenate([obj_1_tmp, np.zeros((1024-obj_1_tmp.shape[0],3))])).unsqueeze(0).cuda()
    #     obj_2_tmp = torch.from_numpy(np.concatenate([obj_2_tmp, np.zeros((1024-obj_2_tmp.shape[0], 3))])).unsqueeze(0).cuda()
    #     obj_1 = torch.from_numpy(np.concatenate([obj_1, np.zeros((1024 - obj_1.shape[0], 3))])).unsqueeze(
    #       0).cuda()
    #     obj_2 = torch.from_numpy(np.concatenate([obj_2, np.zeros((1024 - obj_2.shape[0], 3))])).unsqueeze(
    #       0).cuda()
    #   else:
    #     obj_1_tmp = torch.from_numpy(obj_1_tmp).unsqueeze(0).cuda()
    #     obj_2_tmp = torch.from_numpy(obj_2_tmp).unsqueeze(0).cuda()
    #     obj_1 = torch.from_numpy(obj_1).unsqueeze(0).cuda()
    #     obj_2 = torch.from_numpy(obj_2).unsqueeze(0).cuda()
    #   print(obj_1.shape, obj_2.shape)
    #   emd = emdModule()
    #   dis, assigment = emd(obj_1_tmp, obj_2_tmp, 0.05, 3000)
    #   print("|set(assignment)|: %d" % assigment.unique().numel())
    #   assigment = assigment.cpu().numpy()
    #   assigment = np.expand_dims(assigment, -1)
    #   print(assigment.shape)
    #   obj_2 = np.take_along_axis(obj_2, assigment, axis=1)
    #   obj_compound = 2*(0.5*obj_1 + 0.5*obj_2) + torch.from_numpy(center_obj_1).cuda()
    #   obj_compound = obj_compound.squeeze().cpu().numpy()
    #   obj_aug.append(obj_compound)
    #   idx_final[self.inst_label_valid == inst_idx_pair[i,0]] = 0
    #
    # self.points_valid = self.points_valid[idx_final]
    # obj_aug = np.concatenate(obj_aug)
    # print(obj_aug.shape)
    # self.points_valid = np.concatenate([self.points_valid, obj_aug], axis=0)

    self.set_points(points=self.points_valid)
    self.sem_label = self.sem_label_valid
    self.inst_label = self.inst_label_valid

    # zeros_aux = np.zeros(obj_aug.shape[0]).astype(np.int)
    #
    # self.sem_label = self.sem_label_valid[idx_final]
    # self.inst_label = self.inst_label_valid[idx_final]
    #
    # self.sem_label = np.concatenate([self.sem_label, zeros_aux])
    # self.inst_label = np.concatenate([self.inst_label, zeros_aux])

    # cls, cnt = np.unique(self.sem_label_valid, return_counts = True)
    # print(cls, cnt)
    # cls, cnt = np.unique(self.inst_label_valid, return_counts = True)
    # print(cls, cnt)

# class emdFunction(Function):
#   @staticmethod
#   def forward(ctx, xyz1, xyz2, eps, iters):
#     batchsize, n, _ = xyz1.size()
#     _, m, _ = xyz2.size()
#
#     assert (n == m)
#     assert (xyz1.size()[0] == xyz2.size()[0])
#     assert (n % 1024 == 0)
#     assert (batchsize <= 512)
#
#     xyz1 = xyz1.contiguous().float().cuda()
#     xyz2 = xyz2.contiguous().float().cuda()
#     dist = torch.zeros(batchsize, n, device='cuda').contiguous()
#     assignment = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous() - 1
#     assignment_inv = torch.zeros(batchsize, m, device='cuda', dtype=torch.int32).contiguous() - 1
#     price = torch.zeros(batchsize, m, device='cuda').contiguous()
#     bid = torch.zeros(batchsize, n, device='cuda', dtype=torch.int32).contiguous()
#     bid_increments = torch.zeros(batchsize, n, device='cuda').contiguous()
#     max_increments = torch.zeros(batchsize, m, device='cuda').contiguous()
#     unass_idx = torch.zeros(batchsize * n, device='cuda', dtype=torch.int32).contiguous()
#     max_idx = torch.zeros(batchsize * m, device='cuda', dtype=torch.int32).contiguous()
#     unass_cnt = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
#     unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
#     cnt_tmp = torch.zeros(512, dtype=torch.int32, device='cuda').contiguous()
#
#     emd.forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx,
#                 unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters)
#
#     ctx.save_for_backward(xyz1, xyz2, assignment)
#     return dist, assignment
#
#   @staticmethod
#   def backward(ctx, graddist, gradidx):
#     xyz1, xyz2, assignment = ctx.saved_tensors
#     graddist = graddist.contiguous()
#
#     gradxyz1 = torch.zeros(xyz1.size(), device='cuda').contiguous()
#     gradxyz2 = torch.zeros(xyz2.size(), device='cuda').contiguous()
#
#     emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
#     return gradxyz1, gradxyz2, None, None
#
#
# class emdModule(nn.Module):
#   def __init__(self):
#     super(emdModule, self).__init__()
#
#   def forward(self, input1, input2, eps, iters):
#     return emdFunction.apply(input1, input2, eps, iters)