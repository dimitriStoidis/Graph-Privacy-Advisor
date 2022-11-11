import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ggnn import GGNN
from networks.places365CNN import resnet_scene


class GPA(nn.Module):
    def __init__(self, num_class=2,
            ggnn_hidden_channel=2,
            ggnn_output_channel=2, time_step=3,
            attr_num=81, adjacency_matrix=''):
        
        super(GPA, self).__init__()
        self._num_class = num_class
        self._ggnn_hidden_channel = ggnn_hidden_channel
        self._ggnn_output_channel = ggnn_output_channel
        self._time_step = time_step
        self._adjacency_matrix = adjacency_matrix
        self._attr_num = attr_num
        self._graph_num = attr_num + num_class

        self.ggnn = GGNN(hidden_state_channel=self._ggnn_hidden_channel,
            output_channel=self._ggnn_output_channel,
            time_step=self._time_step,
            adjacency_matrix=self._adjacency_matrix,
            num_classes=self._num_class,
            num_objects=self._attr_num)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self._attr_num + 1, self._num_class),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self._num_class, 1)
        )
        self.scene_privacy_layer = nn.Linear(365, 2)
        self.reshape_input = nn.Linear((self._attr_num + 1) * self._num_class, (self._attr_num + 1))
        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.classifier.modules():
            cnt = 0
            if isinstance(m, nn.Linear):
                if cnt == 0:
                    m.weight.data.normal_(0, 0.001)
                else:
                    m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                cnt += 1

    def forward(self, full_im, categories, card, scene):
        batch_size = 1
        # initialise contextual matrix
        contextual = Variable(torch.zeros(batch_size, self._graph_num, self._num_class),
                              requires_grad=False).cuda()
        if card:
            contextual[:, self._num_class:, 0] = 1.  # size: ([bs,2])
            end_idx = 0
            for b in range(batch_size):
                if categories[b, 0].item() < 12:
                    cur_rois_num = categories[b, 0].item()
                else:
                    cur_rois_num = 12
                end_idx += cur_rois_num
                idxs = categories[b, 1:(cur_rois_num + 1)].data.tolist()

                # Uncomment for average object features and object cardinality
                for i in range(cur_rois_num):
                    # if idxs[i] == 0:  # Uncomment for person detector
                    # second column with cardinality info
                    contextual[b, int(idxs[i]) + self._num_class, 1] = idxs.count(idxs[i])
                    contextual[b, int(idxs[i]) + self._num_class, 2:] = 0

        if scene:
            scene_logit = resnet_scene(full_im.unsqueeze(0))
            binary_scene = self.scene_privacy_layer(scene_logit)
            contextual[:, 0: self._num_class, 1] = binary_scene

        model_input = contextual.view(batch_size, -1)
        ggnn_feature = self.ggnn(model_input)
        ggnn_feature_norm = ggnn_feature.view(batch_size * self._num_class, -1)
        ggnn_feature_norm = self.reshape_input(ggnn_feature_norm)

        final_scores = self.classifier(ggnn_feature_norm).view(batch_size, -1)

        return final_scores
