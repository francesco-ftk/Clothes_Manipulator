""" Clothes Manipulator Task NTM model. """

from attr import attrs, attrib, Factory
from torch import nn
from torch import optim
from MAN_ClothesManipulator.ntm.aio import EncapsulatedNTM


@attrs
class ClothesManipulatroTaskParams(object):
    name = attrib(default="copy-task")
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=1, converter=int)
    num_heads = attrib(default=1, converter=int)  # ok
    manip_vector_width = attrib(default=151, converter=int)  # input
    sequence_output_width = attrib(default=340, converter=int)  # output
    # sequence_min_len = attrib(default=1, converter=int)
    # sequence_max_len = attrib(default=20, converter=int)
    memory_n = attrib(default=12, converter=int)  # memory
    memory_m = attrib(default=340, converter=int)  # memory
    # num_batches = attrib(default=50000, converter=int)
    # batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=1e-4, converter=float)  # optimizer
    rmsprop_momentum = attrib(default=0.9, converter=float)  # optimizer
    rmsprop_alpha = attrib(default=0.95, converter=float)  # optimizer


@attrs
class ClothesManipulatorModelTraining(object):
    params = attrib(default=Factory(ClothesManipulatroTaskParams))
    net = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        net = EncapsulatedNTM(self.params.manip_vector_width, self.params.sequence_output_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @criterion.default
    def default_criterion(self):
        return nn.SmoothL1Loss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
