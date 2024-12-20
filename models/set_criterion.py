import train
from models.matcher import build_matcher
from util.misc import sigmoid_focal_loss, accuracy


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, focal_alpha, focal_gamma, weight_dict):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.coord_criterion = nn.MSELoss(reduction='sum')
        self.class_criterion = nn.CrossEntropyLoss(reduction='sum')
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.weight_dict = weight_dict

    def loss_labels_focal(self, outputs, targets, indices, num_objects, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs

        # [batch_size, number_queries, number_of_classes]
        src_logits = outputs['pred_logits']

        # (batch_ids, output_query_ids)
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # [batch_size, number_queries]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=train.device)

        target_classes[idx] = target_classes_o

        # [batch_size, number_queries, number_of_classes+1]
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=train.device)

        #
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # [batch_size, number_queries, number_of_classes]
        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_objects,
            alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_ce *= src_logits.shape[1] # Why?
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_center_points(self, outputs, targets, indices, num_objects, log=True):
        """L1 center point loss
        targets dicts must contain the key "center_points" containing a tensor of dim [nb_target_boxes, 2]
        """
        assert 'pred_center_points' in outputs

        idx = self._get_src_permutation_idx(indices)
        src_cps = outputs['pred_center_points'][idx]
        target_cps = torch.cat([t['center_points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_cp = F.l1_loss(src_cps, target_cps, reduction='none')

        losses = {}
        losses['loss_center_point'] = loss_cp.sum() / num_objects

        return losses

    def forward(self, outputs, targets):
        indecies = self.matcher(outputs, targets)

        num_objects = sum(len(t["labels"]) for t in targets)
        num_objects = torch.as_tensor(
            [num_objects], dtype=torch.float, device=train.device)

        losses = {}
        loss_labels = self.loss_labels_focal(outputs, targets, indecies, num_objects)
        loss_center_points = self.loss_center_points(outputs, targets, indecies, num_objects)

        losses.update(loss_labels)
        losses.update(loss_center_points)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


def build_criterion(num_classes):
    matcher = build_matcher()
    return SetCriterion(
        num_classes,
        matcher,
        focal_alpha=0.25,
        focal_gamma=2,
        weight_dict={
            'loss_ce': 2,
            'loss_center_point': 5
        }
    )
