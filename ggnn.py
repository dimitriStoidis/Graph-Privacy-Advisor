import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
Code inspired from https://github.com/guang-yanng/Image_Privacy and https://github.com/HCPLab-SYSU/SR
"""


class GGNN(nn.Module):
	def __init__(self, hidden_state_channel=10, output_channel=5,
				time_step=3, adjacency_matrix='', num_classes=2, num_objects=81):
		super(GGNN, self).__init__()

		self.time_step = time_step
		self.hidden_state_channel = hidden_state_channel
		self.output_channel = output_channel
		self.adjacency_matrix = adjacency_matrix
		self.num_classes = num_classes
		self.num_objects = num_objects
		self.cnt = 0

		self._in_matrix, self._out_matrix = self.load_nodes(self.adjacency_matrix)

		self._mask = Variable(torch.zeros(self.num_classes, self.num_objects), requires_grad=False).cuda()
		tmp = self._in_matrix[0:self.num_classes, self.num_classes:]  # reason in ggnn # same as adjacency matrix shape [2, 81]
		self._mask[np.where(tmp > 0)] = 1  # 1 for valid connections between objects and classes
		self._in_matrix = Variable(torch.from_numpy(self._in_matrix), requires_grad=False).cuda() # does this mean that the adjacency matrix is not updated during training?
		self._out_matrix = Variable(torch.from_numpy(self._out_matrix), requires_grad=False).cuda()

		self.fc_eq3_w = nn.Linear(2*hidden_state_channel, hidden_state_channel)
		self.fc_eq3_u = nn.Linear(hidden_state_channel, hidden_state_channel)
		self.fc_eq4_w = nn.Linear(2*hidden_state_channel, hidden_state_channel)
		self.fc_eq4_u = nn.Linear(hidden_state_channel, hidden_state_channel)
		self.fc_eq5_w = nn.Linear(2*hidden_state_channel, hidden_state_channel)
		self.fc_eq5_u = nn.Linear(hidden_state_channel, hidden_state_channel)

		self.fc_output = nn.Linear(2*hidden_state_channel, output_channel)
		self.ReLU = nn.ReLU(True)

		self.reason_fc_x = nn.Linear(hidden_state_channel, output_channel)
		self.reason_fc_y = nn.Linear(hidden_state_channel, output_channel)
		self.reason_fc2 = nn.Linear(output_channel, 1)
	
		self._initialize_weights()

	def forward(self, input):
		batch_size = input.size()[0]

		input = input.view(-1, self.hidden_state_channel)
		#
		node_num = self._in_matrix.size()[0]
		batch_aog_nodes = input.view(-1, node_num, self.hidden_state_channel)
		batch_in_matrix = self._in_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)
		batch_out_matrix = self._out_matrix.repeat(batch_size, 1).view(batch_size, node_num, -1)

		# propogation process
		for t in range(self.time_step):

			# eq(2)
			av1 = torch.cat((torch.bmm(batch_in_matrix, batch_aog_nodes), torch.bmm(batch_out_matrix, batch_aog_nodes)), 2) # shape = [bs, 83, 8196]
			av = av1.view(batch_size * node_num, -1)

			# eq(3) zv = sigma(Wav + Uhv)
			flatten_aog_nodes = batch_aog_nodes.view(batch_size * node_num, -1)
			zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes))

			# eq(4) rv = sigma(Wav + Uhv)
			rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_aog_nodes))

			# eq(5)
			# hv = tanh(Wav + U(rv*hv))
			hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_aog_nodes))
			# hv = (1-zv) * hv + zv * hv
			flatten_aog_nodes = (1 - zv) * flatten_aog_nodes + zv * hv  # shape = [bs*83, 4098]
			batch_aog_nodes = flatten_aog_nodes.view(batch_size, node_num, -1)  # shape = [bs, 83, 4098]

		# final hidden state of all nodes {h1, h2, .., hv}
		output = torch.cat((flatten_aog_nodes, input), 1)

		# node-level feature ov = o([hv, xv]) through fc layer
		output = self.fc_output(output)
		output = torch.tanh(output)  # shape = [bs*83, 2], previously -->  [bs*83, 512]
		# ### reasoning - Attention ###

		fan = flatten_aog_nodes.view(batch_size, node_num, -1)
		num_objects = node_num - self.num_classes  # 81

		rnode = fan[:, 0:self.num_classes, :].contiguous().view(-1, self.hidden_state_channel)
		rfcx = torch.tanh(self.reason_fc_x(rnode))  # shape = [40, 512]
		rnode_enlarge = rfcx.contiguous().view(batch_size * self.num_classes, 1, -1).repeat(1, num_objects, 1)

		onode = fan[:, self.num_classes:, :].contiguous().view(-1, self.hidden_state_channel)
		rfcy = torch.tanh(self.reason_fc_y(onode))  # shape = [1620, 512]
		onode_enlarge = rfcy.contiguous().view(batch_size, 1, num_objects, -1).repeat(1, self.num_classes, 1, 1) # shape = [bs, 2, 81, 512]

		# fuse class and objects hidden states with low-rank bilinear pooling
		rocat = (rnode_enlarge.contiguous().view(-1, self.output_channel)) * (onode_enlarge.contiguous().view(-1, self.output_channel))

		# attention coefficient eij = rfc2
		rfc2_1 = self.reason_fc2(rocat)  # [3240, 1]
		# normalise coefficients with sigmoid -- aij = sigma(eij)
		rfc2_2 = torch.sigmoid(rfc2_1)
		mask_enlarge = self._mask.repeat(batch_size, 1, 1).view(-1, 1)
		rfc2 = rfc2_2 * mask_enlarge  # [3240, 1]
		
		output = output.contiguous().view(batch_size, node_num, -1) # [bs, 83, 512]
		routput = output[:, 0: self.num_classes, :]  # [bs, 2, 512]
		ooutput = output[:, self.num_classes:, :]  # [bs, 81, 512]

		ooutput_enlarge = ooutput.contiguous().view(batch_size, 1, -1).repeat(1, self.num_classes, 1).view(-1, self.output_channel)  # [3240, 512]

		# aggregate norm. attention coeffs. aij and output features Ooi
		# weight features of object nodes (context nodes)
		weight_ooutput = ooutput_enlarge * rfc2  # [3240, 512]
		weight_ooutput = weight_ooutput.view(batch_size, self.num_classes, num_objects, -1)  # [bs, 2, 81, 512]

		# f = [Ori, ai1Oo1, ai2Oo2, ..., aiNOoN]
		final_output = torch.cat((routput.contiguous().view(batch_size, self.num_classes, 1, -1), weight_ooutput), 2)  # [bs, 2, 82, 512] eq. 10
		return final_output

	def _initialize_weights(self):
		for m in self.reason_fc2.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.1)
				m.bias.data.zero_()
		for m in self.reason_fc_x.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
		for m in self.reason_fc_y.modules():
			if isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def load_nodes(self, file):
		mat = np.load(file)
		d_row = mat.shape[0]
		d_col = 2
		in_matrix = np.zeros((d_row + d_col, d_row + d_col))
		in_matrix[:d_row, d_col:] = mat  # adjacency matrix is in the top right corner
		out_matrix = np.zeros((d_row + d_col, d_row + d_col))
		out_matrix[d_col:, :d_row] = mat.transpose()  # adjacency matrix transpose is in the bottom left corner
		return in_matrix.astype(np.float32), out_matrix.astype(np.float32)
