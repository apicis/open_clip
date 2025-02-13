from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist, Tensor

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
                                   F.cross_entropy(logits_per_image, labels) +
                                   F.cross_entropy(logits_per_text, labels)
                           ) / 2

        distill_loss = (
                               self.dist_loss(dist_logits_per_image, logits_per_image) +
                               self.dist_loss(dist_logits_per_text, logits_per_text)
                       ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss


class NegativeLearningLossRandomSample(nn.Module):
    def __init__(
            self,
            reduction:str,
            tokens_num:int,
            token_factor:int
    ) -> None:
        super().__init__()
        self.tokens_num = tokens_num
        self.token_factor = token_factor
        self.nll_loss = nn.NLLLoss(reduction=reduction)

    def get_probabilities_considered_tokens(self, model_probs: Tensor, considered_tokens: Tensor) -> Tensor:
        """ Selects the probabilities of the considered tokens

            B: batch
            V: number of vocabulary tokens
            S: sentence (tokens)
            T: number of sampled tokens

        Args:
            model_probs: probabilities of the model prediction [B x S x V ]
            considered_tokens: considered tokens sampled [B x S x T ]

        Returns:
            probs: probabilities of the considered tokens [B x S x T ]
        """
        # Get considered tokens shape
        b, s, tokens_num = considered_tokens.shape

        # Create mask in considered token indices
        considered_tokens_mask = torch.zeros_like(model_probs, device=model_probs.device, dtype=torch.int64).scatter_(-1,
                                                                                                                    considered_tokens,
                                                                                                                    1.) > 0
        # Extract probabilities of the considered tokens
        probs = model_probs[considered_tokens_mask].view(b, s, tokens_num)
        return probs

    def randomly_sample_from_set(self, tokens_num: int, tokens_set: Tensor) -> Tensor:
        """ Randomly sample tokens_num tokens from tokens_set

            V: number of vocabulary tokens
            S: sentence (tokens)
            T: number of tokens to sample

        Args:
            tokens_num: number of tokens to randomly sample
            tokens_set: set of tokens to sample randomly from [S x V]

        Returns:
            random_tokens: tokens sampled randomly [S x T]
        """
        # Get shape of the token set
        s, v = tokens_set.shape

        # Randomly sample the tokens from the token_set
        random_tokens_ind = torch.zeros(size=[s, tokens_num], dtype=torch.int64, device=tokens_set.device)
        src_tensor = torch.ones_like(tokens_set)
        for token in range(s):
            random_tokens_ind[token] = torch.multinomial(torch.ones(v), num_samples=tokens_num, replacement=False)

        # Create random tokens mask
        random_tokens_mask = torch.zeros_like(tokens_set, device=tokens_set.device).scatter_(dim=-1, index=random_tokens_ind,
                                                                                             src=src_tensor) > 0

        # Extract tokens from token set
        random_tokens = tokens_set[random_tokens_mask].view(s, tokens_num)
        return random_tokens

    def sample_unshared_tokens_descending(self, model_logits: Tensor, target_tokens: Tensor, tokens_num: int,
                                          token_factor: int) -> Tensor:
        """ Samples a set of tokens_num from the set of tokens_num*tokens_factor that have highest logit values (hence probability).

            B: batch
            V: number of vocabulary tokens
            S: sentence (tokens)
            T: number of tokens to sample

        Args:
            model_logits: logits of the model prediction [B x S x V ]
            target_tokens: tokens in the target tensor [B x S]
            tokens_num: number of tokens to randomly sample
            token_factor: multiplying factor of random tokens to enlarge the set to sample from

        Returns:
            sampled_tokens: tokens randomly sampled among the tokens with highest logit values [B x S x T]
        """
        # Get logits shape
        b, s, v = model_logits.shape

        # Initialize tensors
        sampled_tokens = torch.zeros([b, s, tokens_num], dtype=torch.int64, device=model_logits.device)
        logits_temp = model_logits.clone()

        for batch in range(b):
            # Get unique tokens in target
            unique_target_tokens = torch.unique(target_tokens[batch].flatten())

            # Create the mask of target tokens
            target_mask = torch.zeros(size=[s, v], dtype=torch.bool, device=model_logits.device)
            target_mask[:, unique_target_tokens] = True

            # Discard logits of token mask
            logits_temp[batch, target_mask] = -torch.inf

            # Sort logits decreasing and get first indices (tokens)
            desc_tokens_subset = torch.sort(logits_temp[batch], dim=-1, descending=True)[1][:,
                                 0:tokens_num * token_factor]

            # Randomly sample from the set of tokens having highest logit values
            sampled_tokens[batch] = self.randomly_sample_from_set(tokens_num, tokens_set=desc_tokens_subset)
        return sampled_tokens

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Get probabilities
        model_probs = F.softmax(inputs, dim=-1)

        # Sample token randomly
        most_probable_tokens = self.sample_unshared_tokens_descending(model_logits=inputs,
                                                                      target_tokens=targets,
                                                                      tokens_num=self.tokens_num,
                                                                      token_factor=self.token_factor)

        # Get probabilities for those tokens
        sampled_probs = self.get_probabilities_considered_tokens(model_probs, most_probable_tokens)

        # Compute loss for random token
        loss_random = - torch.sum(torch.log(1 - sampled_probs))

        return loss_random


class PositiveNegativeCoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            negative_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.negative_loss = NegativeLearningLossRandomSample(reduction="mean", tokens_num=8000, token_factor=4)

    def forward(self, image_features, text_features, logits, labels, logit_scale, targets=None,
                output_dict=False):

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        negative_loss = self.negative_loss(
            logits,
            targets,
        )
        negative_loss = negative_loss * self.negative_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss, "negative_loss": negative_loss}

        return clip_loss, caption_loss, negative_loss


class TripletLoss(nn.Module):
    def __init__(
            self,
            reduction: str
    ) -> None:
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(reduction=reduction)

    def forward(self, anchors: Tensor, positives: Tensor, negatives:Tensor) -> Tensor:
        # Compute loss
        loss = self.triplet_loss(anchor=anchors, positive=positives, negative=negatives)
        return loss


class TripletCoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            triplet_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(reduction="mean")
        self.triplet_loss = TripletLoss(reduction="mean")

    def forward(self, image_features, text_features, logits, labels, logit_scale, targets=None,
                output_dict=False):
        # Get only anchor examples to compute captioning loss
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[::3] = True
        anchor_logits = logits[mask, :, :]
        anchor_labels = labels[mask, :]
        anchor_image_features = image_features[mask]
        anchor_text_features = text_features[mask]

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(anchor_image_features, anchor_text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            anchor_logits.permute(0, 2, 1),
            anchor_labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        # Separate into anchors, positives and negatives
        mask = torch.zeros(anchor_image_features.shape[0], dtype=torch.bool, device=logits.device)
        mask[::3] = True
        anchors = anchor_image_features[mask]
        mask = torch.zeros(anchor_image_features.shape[0], dtype=torch.bool, device=logits.device)
        mask[1::3] = True
        positives = anchor_image_features[mask]
        mask = torch.zeros(anchor_image_features.shape[0], dtype=torch.bool, device=logits.device)
        mask[2::3] = True
        negatives = anchor_image_features[mask]
        triplet_loss = self.triplet_loss(
            anchors=anchors,
            positives=positives,
            negatives=negatives
        )
        triplet_loss = triplet_loss * self.triplet_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss, "triplet_loss": triplet_loss}

        return clip_loss, caption_loss, triplet_loss

class PositiveNegativeTripletLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            triplet_loss_weight,
            negative_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.triplet_loss = TripletLoss(reduction="mean")
        self.caption_loss = nn.CrossEntropyLoss(reduction="mean")
        self.negative_loss = NegativeLearningLossRandomSample(reduction="mean", tokens_num=1000, token_factor=32)

    def forward(self, image_features, text_features, logits, labels, logit_scale, targets=None,
                output_dict=False):

        # Get only anchor examples to compute captioning loss
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[::3] = True
        anchor_logits = logits[mask, :, :]
        anchor_labels = labels[mask, :]
        anchor_image_features = image_features[mask]
        anchor_text_features = text_features[mask]

        clip_loss = torch.tensor(0)

        if self.clip_loss_weight:
            clip_loss = super().forward(anchor_image_features, anchor_text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            anchor_logits.permute(0, 2, 1),
            anchor_labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        negative_loss = self.negative_loss(
            anchor_logits,
            anchor_labels,
        )
        negative_loss = negative_loss * self.negative_loss_weight

        # Separate into anchors, positives and negatives
        mask = torch.zeros(anchor_image_features.shape[0], dtype=torch.bool, device=logits.device)
        mask[::3] = True
        anchors = anchor_image_features[mask]
        mask = torch.zeros(anchor_image_features.shape[0], dtype=torch.bool, device=logits.device)
        mask[1::3] = True
        positives = anchor_image_features[mask]
        mask = torch.zeros(anchor_image_features.shape[0], dtype=torch.bool, device=logits.device)
        mask[2::3] = True
        negatives = anchor_image_features[mask]
        triplet_loss = self.triplet_loss(
            anchors=anchors,
            positives=positives,
            negatives=negatives
        )
        triplet_loss = triplet_loss * self.triplet_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss, "negative_loss": negative_loss, "triplet_loss": triplet_loss}

        return  clip_loss, caption_loss, negative_loss, triplet_loss
