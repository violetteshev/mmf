# Copyright (c) Facebook, Inc. and its affiliates.
"""
Losses module contains implementations for various losses used generally
in vision and language space. One can register custom losses to be detected by
MMF using the following example.

.. code::

   from mmf.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_config:
       some_model:
           losses:
               - type: custom
               - params: {}
"""
import collections
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmf.common.registry import registry
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_config`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (ListConfig): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instanttions of each loss
                                   passed in config
    """

    def __init__(self, loss_list):
        super().__init__()
        self.losses = nn.ModuleList()
        config = registry.get("config")
        self._evaluation_predict = False
        if config:
            self._evaluation_predict = config.get("evaluation", {}).get(
                "predict", False
            )

        for loss in loss_list:
            self.losses.append(MMFLoss(loss))

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if "targets" not in sample_list:
            if not self._evaluation_predict:
                warnings.warn(
                    "Sample list has not field 'targets', are you "
                    "sure that your ImDB has labels? you may have "
                    "wanted to run with evaluation.predict=true"
                )
            return output

        for loss in self.losses:
            output.update(loss(sample_list, model_output))

        if not torch.jit.is_scripting():
            registry_loss_key = "{}.{}.{}".format(
                "losses", sample_list["dataset_name"], sample_list["dataset_type"]
            )
            # Register the losses to registry
            registry.register(registry_loss_key, output)

        return output


class MMFLoss(nn.Module):
    """Internal MMF helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/vqa2/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set of dataset `vqa2`.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``MMFLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}

        is_mapping = isinstance(params, collections.abc.MutableMapping)

        if is_mapping:
            if "type" not in params:
                raise ValueError(
                    "Parameters to loss must have 'type' field to"
                    "specify type of loss to instantiate"
                )
            else:
                loss_name = params["type"]
        else:
            assert isinstance(
                params, str
            ), "loss must be a string or dictionary with 'type' key"
            loss_name = params

        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        if loss_class is None:
            raise ValueError(f"No loss named {loss_name} is registered to registry")
        # Special case of multi as it requires an array
        if loss_name == "multi":
            assert is_mapping
            self.loss_criterion = loss_class(params)
        else:
            if is_mapping:
                loss_params = params.get("params", {})
            else:
                loss_params = {}
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        loss = self.loss_criterion(sample_list, model_output)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        if not torch.jit.is_scripting():
            key = "{}/{}/{}".format(
                sample_list.dataset_type, sample_list.dataset_name, self.name
            )
        else:
            key = f"{self.name}"
        return {key: loss}


@registry.register_loss("logit_bce")
class LogitBinaryCrossEntropy(nn.Module):
    """Returns Binary Cross Entropy for logits.

    Attention:
        `Key`: logit_bce
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("triple_logit_bce")
class TripleLogitBinaryCrossEntropy(nn.Module):
    """
    This is used for Three-branch fusion only. We predict scores and compute
    cross entropy loss for each of branches.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy for logits
        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.
        Returns:
            torch.FloatTensor: Float value for loss.
        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        if scores.dim() == 3:
            loss = (
                F.binary_cross_entropy_with_logits(
                    scores[:, 0], targets, reduction="mean"
                )
                + F.binary_cross_entropy_with_logits(
                    scores[:, 1], targets, reduction="mean"
                )
                + F.binary_cross_entropy_with_logits(
                    scores[:, 2], targets, reduction="mean"
                )
            )
        else:
            loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")

        return loss * targets.size(-1)


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        loss = F.binary_cross_entropy(scores, targets, reduction="mean")

        return loss * targets.size(1)


@registry.register_loss("caption_cross_entropy")
class CaptionCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the cross entropy loss for captions.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]

        # If no captions(test dataset) then assume decode length to be uniform
        if hasattr(sample_list, "caption_len"):
            caption_lengths, _ = sample_list.caption_len.sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
        else:
            decode_lengths = [targets.size(1)] * targets.size(0)
        if torch.__version__ >= "1.1":
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data
        else:
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = F.cross_entropy(scores, targets)

        return loss


@registry.register_loss("nll_loss")
class NLLLoss(nn.Module):
    """Negative log likelikehood loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the negative log likelihood.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        _, idx = targets.max(dim=1)
        loss = F.nll_loss(scores, idx, reduction="mean")

        return loss * targets.size(1)


def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


@registry.register_loss("multi")
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        params (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: logit_bce
            weight: 0.3
            params: {}
          - type: attention_supervision
            weight: 0.7
            params: {}

    """

    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []

        self.loss_names = []

        for loss_params in params["params"]:
            self.loss_names.append(loss_params["type"])
            loss_fn = MMFLoss(loss_params)
            loss_weight = loss_params.get("weight", {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            value = loss_fn(sample_list, model_output, *args, **kwargs)
            loss += self.losses_weights[idx] * value
        return loss


@registry.register_loss("attention_supervision")
class AttentionSupervisionLoss(nn.Module):
    """Loss for attention supervision. Used in case you want to make attentions
    similar to some particular values.
    """

    def __init__(self):
        super().__init__()
        self.loss_fn = lambda *args, **kwargs: nn.functional.binary_cross_entropy(
            *args, **kwargs
        )

    def forward(self, sample_list, model_output):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        context_attentions = model_output["attentions"]
        attention_supervision = sample_list["info"]["attention_supervision"]

        loss = self.loss_fn(
            context_attentions[0],
            attention_supervision.float(),
            weight=attention_supervision.float(),
        )

        # Multiply average loss back with target size to get actual loss
        return loss * attention_supervision.size(1)


@registry.register_loss("weighted_softmax")
class WeightedSoftmaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = loss * tar_sum
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("softmax_kldiv")
class SoftmaxKlDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = kl_div(res, tar)
        loss = torch.sum(loss) / loss.size(0)
        return loss


@registry.register_loss("wrong")
class WrongLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss = F.kl_div(res, tar, reduction="mean")
        loss *= target_score.size(1)
        return loss


@registry.register_loss("bce_kl_combined")
class CombinedLoss(nn.Module):
    def __init__(self, weight_softmax):
        super().__init__()
        self.weight_softmax = weight_softmax

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(
            pred_score, target_score, reduction="mean"
        )
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        return loss


@registry.register_loss("m4c_decoding_bce_with_mask")
class M4CDecodingBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.0])

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss_mask = sample_list["train_loss_mask"]
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits(scores, targets, reduction="none")
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss


@registry.register_loss("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.CrossEntropyLoss(**params)

    def forward(self, sample_list, model_output):
        return self.loss_fn(model_output["scores"], sample_list.targets)


@registry.register_loss("img_feat_triplet")
class ImgFeatTripletLoss(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.TripletMarginLoss(**params, reduction="sum")

    def forward(self, sample_list, model_output):
        # TODO: implement
        loss = 0.0

        return loss


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i])
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * 10.0

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i])
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, 4.0)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(5.0).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = torch.cat(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.BoolTensor(masks)
        masks = masks.cuda()

    similarities = similarities * 10.0
    if class_ids is not None:
        # TODO: UserWarning: masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,please use a mask with dtype torch.bool instead
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


@registry.register_loss("img_feat_hinge")
class ImgFeatHingeLoss(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}
        self.loss_fn = nn.HingeEmbeddingLoss(**params, reduction="sum")

    def forward(self, sample_list, model_output):
        pred_feat = model_output["cnn_code"]
        target_feat = model_output["gt_cnn_code"]
        targets = sample_list["gan_targets"]
        gt_pred_answers = sample_list["pred_answers"]

        # View with correct sizes
        batch_size = target_feat.size(0)
        cap_num = pred_feat.size(0) // batch_size
        target_feat_expand = target_feat.unsqueeze(dim=1).repeat(1, cap_num, 1).view(batch_size*cap_num, -1)
        targets_concat = targets.view(batch_size*cap_num)

        # Find pairwise distance and loss
        dist = F.pairwise_distance(pred_feat, target_feat_expand, p=2.0, eps=1e-06, keepdim=False)
        loss = self.loss_fn(dist, targets_concat) / batch_size

        # Calculate scores
        dist = dist.view(batch_size, cap_num)
        _, min_idx = dist.min(dim=1)
        pred_answers = gt_pred_answers[range(batch_size), min_idx]
        scores = torch.zeros_like(sample_list["targets"])
        scores[range(batch_size), pred_answers] = 1
        model_output["scores"] = scores

        class_ids = sample_list["image_id"].unsqueeze(dim=1).repeat(1, cap_num).view(batch_size*cap_num)
        match_labels = torch.arange(0, batch_size*cap_num, device=class_ids.device)
        cap_lens = sample_list["cap_len"].view(-1).tolist()
        w_loss0, w_loss1, _ = words_loss(model_output["region_features"], model_output["words_embs"],
                                             match_labels, cap_lens,
                                             class_ids, batch_size*cap_num)
        w_loss = (w_loss0 + w_loss1) * 50.0


        s_loss0, s_loss1 = sent_loss(pred_feat, model_output["sent_emb"],
                                        match_labels, class_ids, batch_size*cap_num)
        s_loss = (s_loss0 + s_loss1) * 50.0


        loss += w_loss + s_loss

        return loss
