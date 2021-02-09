import time
import math
import torch 
import torch.nn as nn
from torch import optim
import torchvision
from a5_helper import *
import matplotlib.pyplot as plt


def hello_single_stage_detector():
    print("Hello from single_stage_detector.py!")


def GenerateAnchor(anc, grid):
  """
  Anchor generator.

  Inputs:
  - anc: Tensor of shape (A, 2) giving the shapes of anchor boxes to consider at
    each point in the grid. anc[a] = (w, h) gives the width and height of the
    a'th anchor shape.
  - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
    center of each feature from the backbone feature map. This is the tensor
    returned from GenerateGrid.
  
  Outputs:
  - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
    anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
    centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
    boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
    and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
    corners of the box.
  """
  anchors = None
  ##############################################################################
  # TODO: Given a set of anchor shapes and a grid cell on the activation map,  #
  # generate all the anchor coordinates for each image. Support batch input.   #
  ##############################################################################
  # Replace "pass" statement with your code
  B, H, W, _ = grid.shape
  A = anc.shape[0]
  anchors = torch.zeros((B,A,H,W,4), device=anc.device)
  #print(grid.shape)
  #print(anc.shape)
  grid = grid.unsqueeze(1)
 # print(grid.shape)
  anc = anc.unsqueeze(0).unsqueeze(2).unsqueeze(3)
  #print(anc.shape)
  anchors[:,:,:,:,:2] = grid - anc/2
  anchors[:,:,:,:,2:4] = grid + anc/2


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return anchors


def GenerateProposal(anchors, offsets, method='YOLO'):
  """
  Proposal generator.

  Inputs:
  - anchors: Anchor boxes, of shape (B, A, H', W', 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
  - offsets: Transformations of shape (B, A, H', W', 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[b, a, h, w] = (tx, ty, tw, th) will be applied to the anchor
    anchors[b, a, h, w]. For YOLO, assume that tx and ty are in the range
    (-0.5, 0.5).
  - method: Which transformation formula to use, either 'YOLO' or 'FasterRCNN'
  
  Outputs:
  - proposals: Region proposals of shape (B, A, H', W', 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[b, a, h, w] to the anchor [b, a, h, w] should give the
    proposal proposals[b, a, h, w].
  
  """
  assert(method in ['YOLO', 'FasterRCNN'])
  proposals = None
  ##############################################################################
  # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
  # compute the proposal coordinates using the transformation formulas above.  #
  ##############################################################################
  # Replace "pass" statement with your code
  proposals = torch.zeros_like(anchors)
  anchor_center_hw = torch.zeros_like(anchors) ##Putting coordinates to (center_x,center_y,width,height)
  anchor_center_hw[:,:,:,:,0] = (anchors[:,:,:,:,0] + anchors[:,:,:,:,2])/2
  anchor_center_hw[:,:,:,:,1] = (anchors[:,:,:,:,3] + anchors[:,:,:,:,1])/2
  anchor_center_hw[:,:,:,:,2] = anchors[:,:,:,:,2] - anchors[:,:,:,:,0]
  anchor_center_hw[:,:,:,:,3] = torch.abs(anchors[:,:,:,:,1] - anchors[:,:,:,:,3])

  if method == 'YOLO':
    proposals[:,:,:,:,0] = anchor_center_hw[:,:,:,:,0] + offsets[:,:,:,:,0]
    proposals[:,:,:,:,1] = anchor_center_hw[:,:,:,:,1] + offsets[:,:,:,:,1]
    proposals[:,:,:,:,2] = anchor_center_hw[:,:,:,:,2] * torch.exp(offsets[:,:,:,:,2])
    proposals[:,:,:,:,3] = anchor_center_hw[:,:,:,:,3] * torch.exp(offsets[:,:,:,:,3])

  #### 
  if method == "FasterRCNN":
    proposals[:,:,:,:,0] = anchor_center_hw[:,:,:,:,0] + offsets[:,:,:,:,0]*anchor_center_hw[:,:,:,:,2]
    proposals[:,:,:,:,1] = anchor_center_hw[:,:,:,:,1] + offsets[:,:,:,:,1]*anchor_center_hw[:,:,:,:,3]
    proposals[:,:,:,:,2] = anchor_center_hw[:,:,:,:,2] * torch.exp(offsets[:,:,:,:,2])
    proposals[:,:,:,:,3] = anchor_center_hw[:,:,:,:,3] * torch.exp(offsets[:,:,:,:,3])

  proposals_ = proposals.clone()
  center_x = proposals_[:,:,:,:,0].clone()
  center_y = proposals_[:,:,:,:,1].clone()
  width = proposals_[:,:,:,:,2].clone()
  height = proposals_[:,:,:,:,3].clone()
  proposals[:,:,:,:,2] = center_x + width/2
  proposals[:,:,:,:,0] = center_x - width/2
  proposals[:,:,:,:,3] = center_y + height/2
  proposals[:,:,:,:,1] = center_y - height/2
  ############Limit intermediate calculations...

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return proposals


def IoU(proposals, bboxes):
  """
  Compute intersection over union between sets of bounding boxes.

  Inputs:
  - proposals: Proposals of shape (B, A, H', W', 4)
  - bboxes: Ground-truth boxes from the DataLoader of shape (B, N, 5).
    Each ground-truth box is represented as tuple (x_lr, y_lr, x_rb, y_rb, class).
    If image i has fewer than N boxes, then bboxes[i] will be padded with extra
    rows of -1.
  
  Outputs:
  - iou_mat: IoU matrix of shape (B, A*H'*W', N) where iou_mat[b, i, n] gives
    the IoU between one element of proposals[b] and bboxes[b, n].

  For this implementation you DO NOT need to filter invalid proposals or boxes;
  in particular you don't need any special handling for bboxxes that are padded
  with -1.
  """
  iou_mat = None
  ##############################################################################
  # TODO: Compute the Intersection over Union (IoU) on proposals and GT boxes. #
  # No need to filter invalid proposals/bboxes (i.e., allow region area <= 0). #
  # You need to ensure your implementation is efficient (no for loops).        #
  # HINT:                                                                      #
  # IoU = Area of Intersection / Area of Union, where
  # Area of Union = Area of Proposal + Area of BBox - Area of Intersection     #
  # and the Area of Intersection can be computed using the top-left corner and #
  # bottom-right corner of proposal and bbox. Think about their relationships. #
  ##############################################################################
  # Replace "pass" statement with your code
  # flatten the image from B,A,H,W.4 -> B,A*H*W*4
  B,A,H,W,_ = proposals.shape 
  proposals = proposals.reshape(B,-1,4)
  width = proposals[:,:,2] - proposals[:,:,0] 
  height = proposals[:,:,3] - proposals[:,:,1] 
  area_prop = (width)*(height)
  box_width = bboxes[:,:,2] - bboxes[:,:,0]
  box_height = bboxes[:,:,3] - bboxes[:,:,1]
  area_box = (box_width)*(box_height)
  box = bboxes[:,:,:4].unsqueeze(1)
  prop = proposals.unsqueeze(2)

  tl_x = torch.max(prop[:,:,:,0],box[:,:,:,0])
  tl_y = torch.max(prop[:,:,:,1],box[:,:,:,1])
  br_x = torch.min(prop[:,:,:,2],box[:,:,:,2])
  br_y = torch.min(prop[:,:,:,3],box[:,:,:,3])

  w2 = (br_x - tl_x).clamp(min=0)
  h2 = (br_y - tl_y).clamp(min=0)
  area_prop = torch.unsqueeze(area_prop,2)
  area_box = torch.unsqueeze(area_box,1)
  intersection = w2*h2

  total_area = area_prop + area_box - intersection
  iou_mat = torch.true_divide(intersection,total_area)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return iou_mat


class PredictionNetwork(nn.Module):
  def __init__(self, in_dim, hidden_dim=128, num_anchors=9, num_classes=20, drop_ratio=0.3):
    super().__init__()

    assert(num_classes != 0 and num_anchors != 0)
    self.num_classes = num_classes
    self.num_anchors = num_anchors

    ##############################################################################
    # TODO: Set up a network that will predict outputs for all anchors. This     #
    # network should have a 1x1 convolution with hidden_dim filters, followed    #
    # by a Dropout layer with p=drop_ratio, a Leaky ReLU nonlinearity, and       #
    # finally another 1x1 convolution layer to predict all outputs. You can      #
    # use an nn.Sequential for this network, and store it in a member variable.  #
    # HINT: The output should be of shape (B, 5*A+C, 7, 7), where                #
    # A=self.num_anchors and C=self.num_classes.                                 #
    ##############################################################################
    self.pred_layer = None
    # Replace "pass" statement with your code
    A = self.num_anchors
    C = self.num_classes

    self.pred_layer = nn.Sequential(
                            nn.Conv2d(in_dim,hidden_dim,1),
                            nn.Dropout(drop_ratio),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dim,5*A+C,1))
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  def _extract_anchor_data(self, anchor_data, anchor_idx):
    """
    Inputs:
    - anchor_data: Tensor of shape (B, A, D, H, W) giving a vector of length
      D for each of A anchors at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving anchor indices to extract

    Returns:
    - extracted_anchors: Tensor of shape (M, D) giving anchor data for each
      of the anchors specified by anchor_idx.
    """
    B, A, D, H, W = anchor_data.shape
    anchor_data = anchor_data.permute(0, 1, 3, 4, 2).contiguous().view(-1, D)
    extracted_anchors = anchor_data[anchor_idx]
    return extracted_anchors
  
  def _extract_class_scores(self, all_scores, anchor_idx):
    """
    Inputs:
    - all_scores: Tensor of shape (B, C, H, W) giving classification scores for
      C classes at each point in an H x W grid.
    - anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors at
      which to extract classification scores

    Returns:
    - extracted_scores: Tensor of shape (M, C) giving the classification scores
      for each of the anchors specified by anchor_idx.
    """
    B, C, H, W = all_scores.shape
    A = self.num_anchors
    all_scores = all_scores.contiguous().permute(0, 2, 3, 1).contiguous()
    all_scores = all_scores.view(B, 1, H, W, C).expand(B, A, H, W, C)
    all_scores = all_scores.reshape(B * A * H * W, C)
    extracted_scores = all_scores[anchor_idx]
    return extracted_scores

  def forward(self, features, pos_anchor_idx=None, neg_anchor_idx=None):
    """
    Run the forward pass of the network to predict outputs given features
    from the backbone network.

    Inputs:
    - features: Tensor of shape (B, in_dim, 7, 7) giving image features computed
      by the backbone network.
    - pos_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as positive. These are only given during training; at test-time
      this should be None.
    - neg_anchor_idx: int64 Tensor of shape (M,) giving the indices of anchors
      marked as negative. These are only given at training; at test-time this
      should be None.
    
    The outputs from this method are different during training and inference.
    
    During training, pos_anchor_idx and neg_anchor_idx are given and identify
    which anchors should be positive and negative, and this forward pass needs
    to extract only the predictions for the positive and negative anchors.

    During inference, only features are provided and this method needs to return
    predictions for all anchors.

    Outputs (During training):
    - conf_scores: Tensor of shape (2*M, 1) giving the predicted classification
      scores for positive anchors and negative anchors (in that order).
    - offsets: Tensor of shape (M, 4) giving predicted transformation for
      positive anchors.
    - class_scores: Tensor of shape (M, C) giving classification scores for
      positive anchors.

    Outputs (During inference):
    - conf_scores: Tensor of shape (B, A, H, W) giving predicted classification
      scores for all anchors.
    - offsets: Tensor of shape (B, A, 4, H, W) giving predicted transformations
      all all anchors.
    - class_scores: Tensor of shape (B, C, H, W) giving classification scores for
      each spatial position.
    """
    conf_scores, offsets, class_scores = None, None, None
    ############################################################################
    # TODO: Use backbone features to predict conf_scores, offsets, and         #
    # class_scores. Make sure conf_scores is between 0 and 1 by squashing the  #
    # network output with a sigmoid. Also make sure the first two elements t^x #
    # and t^y of offsets are between -0.5 and 0.5 by squashing with a sigmoid  #
    # and subtracting 0.5.                                                     #
    #                                                                          #
    # During training you need to extract the outputs for only the positive    #
    # and negative anchors as specified above.                                 #
    #                                                                          #
    # HINT: You can use the provided helper methods self._extract_anchor_data  #
    # and self._extract_class_scores to extract information for positive and   #
    # negative anchors specified by pos_anchor_idx and neg_anchor_idx.         #
    ############################################################################
    # Replace "pass" statement with your code
    A = self.num_anchors
    C = self.num_classes
   # M = pos_anchor_idx.shape[0]
    B, in_dim, H, W = features.shape #H,W = 7
    out = self.pred_layer(features)
    class_scores = out[:,5*A:,:,:]
   # print(out.shape)
    offsets = out[:, :5*A, :, :].view(B, A, 5, H, W)[:,:,1:,:,:].clone()
    offsets[:, :, :2, :, :] = torch.sigmoid(offsets[:, :, :2, :, :]) - 0.5 * torch.ones_like(offsets[:, :, :2, :, :])
    conf_scores = out[:, :5*A, :, :].view(B, A, 5, H, W)[:,:,0,:,:].clone().unsqueeze(2)
    conf_scores = torch.sigmoid(conf_scores)
   #print(conf_scores.shape)
    inference = (pos_anchor_idx == None)
    if inference: 
     # print(conf_scores.shape)
      conf_scores = conf_scores.squeeze()
    else:
      #train time
      offsets = self._extract_anchor_data(offsets,pos_anchor_idx)
      class_scores = self._extract_class_scores(class_scores,pos_anchor_idx)
      conf_scores_neg = self._extract_anchor_data(conf_scores,neg_anchor_idx)
      conf_scores_pos = self._extract_anchor_data(conf_scores, pos_anchor_idx)
      conf_scores = torch.cat((conf_scores_pos, conf_scores_neg),dim=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return conf_scores, offsets, class_scores


class SingleStageDetector(nn.Module):
  def __init__(self):
    super().__init__()

    self.anchor_list = torch.tensor([[1., 1], [2, 2], [3, 3], [4, 4], [5, 5], [2, 3], [3, 2], [3, 5], [5, 3]]) # READ ONLY
    self.feat_extractor = FeatureExtractor()
    self.num_classes = 20
    self.pred_network = PredictionNetwork(1280, num_anchors=self.anchor_list.shape[0], \
                                          num_classes=self.num_classes)
  def forward(self, images, bboxes):
    """
    Training-time forward pass for the single-stage detector.

    Inputs:
    - images: Input images, of shape (B, 3, 224, 224)
    - bboxes: GT bounding boxes of shape (B, N, 5) (padded)

    Outputs:
    - total_loss: Torch scalar giving the total loss for the batch.
    """
    # weights to multiple to each loss term
    w_conf = 1 # for conf_scores
    w_reg = 1 # for offsets
    w_cls = 1 # for class_prob

    total_loss = None
    ##############################################################################
    # TODO: Implement the forward pass of SingleStageDetector.                   #
    # A few key steps are outlined as follows:                                   #
    # i) Image feature extraction,                                               #
    # ii) Grid and anchor generation,                                            #
    # iii) Compute IoU between anchors and GT boxes and then determine activated/#
    #      negative anchors, and GT_conf_scores, GT_offsets, GT_class,           #
    # iv) Compute conf_scores, offsets, class_prob through the prediction network#
    # v) Compute the total_loss which is formulated as:                          #
    #    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss,  #
    #    where conf_loss is determined by ConfScoreRegression, w_reg by          #
    #    BboxRegression, and w_cls by ObjectClassification.                      #
    # HINT: Set `neg_thresh=0.2` in ReferenceOnActivatedAnchors in this notebook #
    #       (A5-1) for a better performance than with the default value.         #
    ##############################################################################
    # Replace "pass" statement with your code
    B, N, _ = bboxes.shape

    features = self.feat_extractor(images)
    grid = GenerateGrid(B, w_amap=7, h_amap=7, dtype=torch.float32, device='cuda')
    anchors = GenerateAnchor(self.anchor_list.to(images.device), grid)
    #conf_scores, offsets, class_scores = self.pred_network.forward(features)
   # proposals = GenerateProposal(anchors, offsets)
    iou_mat = IoU(anchors, bboxes)
   # IoU_mat = IoU(proposals, bboxes)

    activated_anch_idx, negative_anch_idx, GT_conf_scores, GT_offsets, GT_class, activated_anc_coord, negative_anc_coord = ReferenceOnActivatedAnchors(anchors, bboxes, grid, iou_mat, neg_thresh=0.2, method='YOLO')
    #proposals = GenerateProposal(anchors, offsets)
    conf_scores, offsets, class_scores = self.pred_network.forward(features, activated_anch_idx, negative_anch_idx)
  
    conf_loss = ConfScoreRegression(conf_scores, GT_conf_scores)
    reg_loss = BboxRegression(offsets, GT_offsets)
    cls_loss = ObjectClassification(class_scores, GT_class, B, torch.prod(torch.tensor(anchors.shape[1:-1])), activated_anch_idx)
    total_loss = w_conf * conf_loss + w_reg * reg_loss + w_cls * cls_loss

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return total_loss
  
  def inference(self, images, thresh=0.5, nms_thresh=0.7):
    """"
    Inference-time forward pass for the single stage detector.

    Inputs:
    - images: Input images
    - thresh: Threshold value on confidence scores
    - nms_thresh: Threshold value on NMS

    Outputs:
    - final_propsals: Keeped proposals after confidence score thresholding and NMS,
                      a list of B (*x4) tensors
    - final_conf_scores: Corresponding confidence scores, a list of B (*x1) tensors
    - final_class: Corresponding class predictions, a list of B  (*x1) tensors
    """
    final_proposals, final_conf_scores, final_class = [], [], []
    ##############################################################################
    # TODO: Predicting the final proposal coordinates `final_proposals`,         #
    # confidence scores `final_conf_scores`, and the class index `final_class`.  #
    # The overall steps are similar to the forward pass but now you do not need  #
    # to decide the activated nor negative anchors.                              #
    # HINT: Thresholding the conf_scores based on the threshold value `thresh`.  #
    # Then, apply NMS (torchvision.ops.nms) to the filtered proposals given the  #
    # threshold `nms_thresh`.                                                    #
    # The class index is determined by the class with the maximal probability.   #
    # Note that `final_propsals`, `final_conf_scores`, and `final_class` are all #
    # lists of B 2-D tensors (you may need to unsqueeze dim=1 for the last two). #
    ##############################################################################
    # Replace "pass" statement with your code
    with torch.no_grad():
      features = self.feat_extractor(images)

      #create anchors..
      B = images.shape[0]
      grid = GenerateGrid(B)
      anchors = GenerateAnchor(self.anchor_list.cuda(), grid)
      
      # prediction network..
      conf_scores, offsets, class_scores = self.pred_network(features)
      B,A,H,W = conf_scores.shape
      num_ancs = A*H*W
    # print('If this is true, set batch_size = B', B==batch_size )
      _,C,_,_ = class_scores.shape  
      offsets = offsets.permute((0,1,3,4,2)) ##B,A,H,W,4
      proposals = GenerateProposal(anchors, offsets, method='YOLO')   ##B,A,H,W,4
      conf_scores = conf_scores.permute((0,2,3,1)).reshape(B,-1) ##(B,HWA)
      proposals = proposals.permute((0,2,3,1,4)).reshape(B,-1,4) ##B,HWA,4
      class_scores = class_scores.permute((0,2,3,1))
      _, max_score_idx = class_scores.max(dim=3)
      max_score_idx = max_score_idx.reshape(B,-1) #B,HW

    for i in range(B):
      # get proposals, confidence scores for i-th image
      curr_conf_scores = conf_scores[i]
      curr_proposals = proposals[i]
      curr_class_scores = max_score_idx[i] #(HW)
      curr_class_scores = curr_class_scores.unsqueeze(1).repeat(1,A).reshape(-1) #(HWA)

      #keeping relelvent conf_scores
      bool_mask = curr_conf_scores > thresh
      curr_conf_scores = curr_conf_scores[bool_mask]
      curr_proposals = curr_proposals[bool_mask,:]
      curr_class_scores = curr_class_scores[bool_mask]
      #removing boxes via nms
      nms_mask = nms(curr_proposals, curr_conf_scores, iou_threshold=nms_thresh)
      
      final_proposals.append(curr_proposals[nms_mask,:])
      final_conf_scores.append(curr_conf_scores[nms_mask].unsqueeze(1))
      final_class.append(curr_class_scores[nms_mask].unsqueeze(1))

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return final_proposals, final_conf_scores, final_class


def nms(boxes, scores, iou_threshold=0.5, topk=None):
  """
  Non-maximum suppression removes overlapping bounding boxes.

  Inputs:
  - boxes: top-left and bottom-right coordinate values of the bounding boxes
    to perform NMS on, of shape Nx4
  - scores: scores for each one of the boxes, of shape N
  - iou_threshold: discards all overlapping boxes with IoU > iou_threshold; float
  - topk: If this is not None, then return only the topk highest-scoring boxes.
    Otherwise if this is None, then return all boxes that pass NMS.

  Outputs:
  - keep: torch.long tensor with the indices of the elements that have been
    kept by NMS, sorted in decreasing order of scores; of shape [num_kept_boxes]
  """

  if (not boxes.numel()) or (not scores.numel()):
    return torch.zeros(0, dtype=torch.long)

  keep = None
  #############################################################################
  # TODO: Implement non-maximum suppression which iterates the following:     #
  #       1. Select the highest-scoring box among the remaining ones,         #
  #          which has not been chosen in this step before                    #
  #       2. Eliminate boxes with IoU > threshold                             #
  #       3. If any boxes remain, GOTO 1                                      #
  #       Your implementation should not depend on a specific device type;    #
  #       you can use the device of the input if necessary.                   #
  # HINT: You can refer to the torchvision library code:                      #
  #   github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/nms_cpu.cpp  #
  #############################################################################
  # Replace "pass" statement with your code
  highest_scores = torch.argsort(scores, descending=True)
  best_boxes = boxes[highest_scores,:] #highest scoring boxes
  box_areas = torch.prod(boxes[:, 2:] - boxes[:, :2], dim=1)
  keep = []
  while len(highest_scores)>0:
      highest_score_idx = highest_scores[0]
      keep.append(highest_score_idx)
      if topk and len(keep)==topk:
          return torch.tensor(keep, device=boxes.device,dtype=torch.int64)

      curr_box = boxes[highest_score_idx]
     # x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
      best_boxes = boxes[highest_scores,:]
      top_lefts = torch.max(curr_box[:2],best_boxes[:,:2])
      bottom_rights = torch.min(curr_box[2:],best_boxes[:,2:])

      intersection = torch.prod(bottom_rights-top_lefts, dim=1)
      intersection *= (top_lefts < bottom_rights).all(dim=1)  #asserts that intersection only holds when they truly overlap..

      curr_area = box_areas[highest_score_idx]
      union = curr_area + box_areas[highest_scores] - intersection
      iou_mat = torch.div(intersection,union).squeeze()
      highest_scores = highest_scores[torch.where(iou_mat <= iou_threshold)]###

  keep = torch.tensor(keep).to(torch.int64).to(boxes.device)

  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  return keep

def ConfScoreRegression(conf_scores, GT_conf_scores):
  """
  Use sum-squared error as in YOLO

  Inputs:
  - conf_scores: Predicted confidence scores
  - GT_conf_scores: GT confidence scores
  
  Outputs:
  - conf_score_loss
  """
  # the target conf_scores for negative samples are zeros
  GT_conf_scores = torch.cat((torch.ones_like(GT_conf_scores), \
                              torch.zeros_like(GT_conf_scores)), dim=0).view(-1, 1)
  conf_score_loss = torch.sum((conf_scores - GT_conf_scores)**2) * 1. / GT_conf_scores.shape[0]
  return conf_score_loss


def BboxRegression(offsets, GT_offsets):
  """"
  Use sum-squared error as in YOLO
  For both xy and wh

  Inputs:
  - offsets: Predicted box offsets
  - GT_offsets: GT box offsets
  
  Outputs:
  - bbox_reg_loss
  """
  bbox_reg_loss = torch.sum((offsets - GT_offsets)**2) * 1. / GT_offsets.shape[0]
  return bbox_reg_loss

