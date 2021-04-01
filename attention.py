import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
	def __init__(self, feat_size, hidden_size, attn_size):
		super(TemporalAttention, self).__init__()
		'''
		Spatial Attention module. It depends on previous hidden memory in the decoder(of shape hidden_size),
		feature at the source side ( of shape(196,feat_size) ).  
		at(s) = align(ht,hs)
			  = exp(score(ht,hs)) / Sum(exp(score(ht,hs')))  
		where
		score(ht,hs) = ht.t * hs                         (dot)
					 = ht.t * Wa * hs                  (general)
					 = va.t * tanh(Wa[ht;hs])           (concat)  
		Here we have used concat formulae.
		Argumets:
		  hidden_size : hidden memory size of decoder.
		  feat_size : feature size of each grid (annotation vector) at encoder side.
		  bottleneck_size : intermediate size.
		'''
		self.hidden_size = hidden_size
		self.feat_size = feat_size
		self.bottleneck_size = attn_size

		self.decoder_projection = nn.Linear(self.hidden_size, self.bottleneck_size)
		self.encoder_projection = nn.Linear(self.feat_size, self.bottleneck_size)
		self.final_projection = nn.Linear(self.bottleneck_size, 1)

	def forward(self, hidden, feats):
		"""
		shape of hidden (hidden_size)
		shape of feats (batch size, feat_len, feat_size)
		"""
		Wh = self.decoder_projection(hidden)
		Uv = self.encoder_projection(feats)
		Wh = Wh.unsqueeze(1).expand_as(Uv)
		energies = self.final_projection(torch.tanh(Wh + Uv))
		weights = F.softmax(energies, dim=1)
		weighted_feats = feats * weights.expand_as(feats)
		attn_feats = weighted_feats.sum(dim=1)
		return attn_feats, weights
