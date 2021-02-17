import math

import torch
from torch.distributions import kl_divergence, Normal

from torch.nn import Module

from prob_utils import normal_parse_params

import random
import torchvision
from torch.nn.functional import softplus, softmax

import torch.nn as nn
from vae_models.model import CResBlock


class VAE(Module):
	"""
	Variational Autoencoder with Arbitrary Conditioning core model.
	It is rather flexible, but have several assumptions:
	+ The batch of objects and the mask of unobserved features
	  have the same shape.
	+ The prior and proposal distributions in the latent space
	  are component-wise independent Gaussians.
	The constructor takes
	+ Prior and proposal network which take as an input the concatenation
	  of the batch of objects and the mask of unobserved features
	  and return the parameters of Gaussians in the latent space.
	  The range of neural networks outputs should not be restricted.
	+ Generative network takes latent representation as an input
	  and returns the parameters of generative distribution
	  p_theta(x_b | z, x_{1 - b}, b), where b is the mask
	  of unobserved features. The information about x_{1 - b} and b
	  can be transmitted to generative network from prior network
	  through nn_utils.MemoryLayer. It is guaranteed that for every batch
	  prior network is always executed before generative network.
	+ Reconstruction log probability. rec_log_prob is a callable
	  which takes (groundtruth, distr_params, mask) as an input
	  and return vector of differentiable log probabilities
	  p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
	+ Sigma_mu and sigma_sigma are the coefficient of the regularization
	  in the hidden space. The default values correspond to a very weak,
	  almost disappearing regularization, which is suitable for all
	  experimental setups the model was tested on.
	"""
	def __init__(self, rec_log_prob, encoder_network, generative_network, channels=3, sigma_mu=1e4, sigma_sigma=1e-4, cond=[], cond_method="resblock"):
		super().__init__()
		self.rec_log_prob = rec_log_prob
		self.encoder_network = encoder_network
		self.generative_network = generative_network
		self.sigma_mu = sigma_mu
		self.sigma_sigma = sigma_sigma
		self.channels = channels

		self.cond = cond
		
		self.class_embed = None

		if cond_method == "input" and cond:
			self.class_embed = nn.Sequential(
				nn.Linear(len(cond), 32),
				nn.BatchNorm1d(32),
				nn.ReLU(),
				nn.Linear(32, 32),
				nn.BatchNorm1d(32),
				nn.ReLU(),
				nn.Linear(32, 256 * 256 * channels)
			)
		self.cond_method = cond_method            

	def make_latent_distributions(self, latent_params, no_proposal=False):
		"""
		Make latent distributions for the given batch and mask.
		No no_proposal is True, return None instead of proposal distribution.
		"""

		# print(latent_params.shape, "=================")
		latent = normal_parse_params(latent_params, 1e-3)

		return latent


	# def prior_regularization(self, prior):
	#     """
	#     The prior distribution regularization in the latent space.
	#     Though it saves prior distribution parameters from going to infinity,
	#     the model usually doesn't diverge even without this regularization.
	#     It almost doesn't affect learning process near zero with default
	#     regularization parameters which are recommended to be used.
	#     """
	#     num_objects = prior.mean.shape[0]
	#     mu = prior.mean.view(num_objects, -1)
	#     sigma = prior.scale.view(num_objects, -1)
	#     mu_regularizer = -(mu ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
	#     sigma_regularizer = (sigma.log() - sigma).sum(-1) * self.sigma_sigma

	#     return mu_regularizer + sigma_regularizer

	def encode(self, batch, metadata):
		
		if self.cond:
			if self.cond_method == "input":

				# if metadata is not None and metadata.shape[1] > 0:

				# metadata_input = metadata.unsqueeze(2).unsqueeze(3).repeat(1, 1, batch.shape[2], batch.shape[3])
				metadata_input = self.class_embed(metadata).reshape(-1, self.channels, 256, 256)
				input_batch = torch.cat((batch, metadata_input), 1).float()

				latent_params = self.encoder_network(input_batch)

			elif self.cond_method == "resblock":
				out = batch
				for layer in self.encoder_network:

					if isinstance(layer, CResBlock):
						out = layer(out, metadata)

					else:

						out = layer(out)


				latent_params = out

		else:
			latent_params = self.encoder_network(batch)

		return latent_params

		
	def decode(self, latent, metadata_latent):

		if self.cond:
			if self.cond_method == "input":
			# if metadata is not None and metadata.shape[1] > 0:
				metadata_latent = metadata_latent.unsqueeze(2).unsqueeze(3).repeat(1, 1, latent.shape[2], latent.shape[3])
				latent = torch.cat((latent, metadata_latent), 1).float()#

				rec_params = self.generative_network(latent)

			elif self.cond_method == "resblock":
				out = latent
				for layer in self.generative_network:
					if isinstance(layer, CResBlock):
						out = layer(out, metadata_latent)

					else:
						out = layer(out)


				rec_params = out
		else:
			rec_params = self.generative_network(latent)
			
		return rec_params




	def batch_vlb(self, batch, middle_mask=False, beta=1, metadata=None):
		"""
		Compute differentiable lower bound for the given batch of objects
		and mask.
		"""

		# input_batch = batch
		# print(metadata)


		latent_params = self.encode(batch, metadata)
		latent_dist = self.make_latent_distributions(latent_params)

		latent = latent_dist.rsample()

		
		# rec_params = self.generative_network(latent)
		
		rec_params = self.decode(latent, metadata)

		rec_loss_t = self.rec_log_prob(batch, rec_params, reduction="none")
		rec_loss = rec_loss_t.sum(-1).sum(-1).sum(-1)
		kl = kl_divergence(latent_dist, Normal(0, 1)).view(batch.shape[0], -1).sum(-1)

		
		# print(rec_loss.shape, kl.shape, "====")
		return rec_loss + beta * kl, rec_loss_t, kl, rec_params


	# def features(self, batch, middle_mask=False, K=1, latent_means=False, one_tailed=False, anomaly_score="logp", metadata=None):
	#     """
	#     Compute differentiable lower bound for the given batch of objects
	#     and mask.
	#     """

	#     samples_params = []
	#     input_batch = batch
	#     if metadata is not None:
	#         metadata_input = metadata.unsqueeze(2).unsqueeze(3).repeat(1, 1, batch.shape[2], batch.shape[3])
	#         input_batch = torch.cat((batch, metadata_input), 1).float()


	#     # if middle_mask:
	#     #     latent_dist = self.make_latent_distributions(batch[:, [0, 2]])
		
	#     # else:
	#     latent_dist = self.make_latent_distributions(input_batch)
	#     kl = kl_divergence(latent_dist, Normal(0, 1)).view(batch.shape[0], -1).mean(-1)

	#     # print(one_tailed, anomaly_score)
	#     total_rec_loss = 0

	#     for i in range(K):
	#         if latent_means:
	#             latent = latent_dist.mean
	#         else:
	#             latent = latent_dist.rsample()

	#         if metadata is not None:
	#             metadata_latent = metadata.unsqueeze(2).unsqueeze(3).repeat(1, 1, latent.shape[2], latent.shape[3])
	#             latent = torch.cat((latent, metadata_latent), 1).float()#

	#         sample_params = self.generative_network(latent)



	#         # if middle_mask:
	#         #     rec_loss = self.rec_log_prob(batch[:, 1:2], sample_params, reduction="none", one_tailed=one_tailed)
	#         # else:
	#         if anomaly_score == "logp":   
	#             rec_loss = self.rec_log_prob(batch, sample_params, reduction="none")
	#         elif anomaly_score == "l1":
	#             rec_loss = -self.rec_log_prob(batch, sample_params.clamp(0, 1), reduction="none")

	#         elif anomaly_score == "std":
	#             rec_loss = (batch - sample_params[:, :3]) / (softplus(sample_params[:, 3:]).clamp(min=0.01))
	#             if one_tailed:
	#                 rec_loss = rec_loss.clamp(min=0)

	#             print("std")
	#         # kl = kl_divergence(latent_dist, Normal(0, 1)).view(batch.shape[0], -1).mean(-1)
		
	#         total_rec_loss += rec_loss

	#         samples_params.append(sample_params.unsqueeze(1))
		


	#     prior_regularization = 0
	#     return total_rec_loss, kl, prior_regularization, torch.cat(samples_params, 1)


	
	def batch_iwae(self, batch, m, metadata, K, middle_mask=False):
		"""
		Compute IWAE log likelihood estimate with K samples per object.
		Technically, it is differentiable, but it is recommended to use it
		for evaluation purposes inside torch.no_grad in order to save memory.
		With torch.no_grad the method almost doesn't require extra memory
		for very large K.
		The method makes K independent passes through generator network,
		so the batch size is the same as for training with batch_vlb.
		"""
		# input_batch = batch
		latent_params = self.encode(batch, metadata)
		latent_dist = self.make_latent_distributions(latent_params)

		kl = kl_divergence(latent_dist, Normal(0, 1)).view(batch.shape[0], -1).sum(-1)

		estimates = []

		e_sum = 0
		for i in range(K):

			# latent = latent_dist.rsample()
			latent = latent_dist.mean


			rec_params = self.decode(latent, metadata)
			
			rec_loss = self.rec_log_prob(batch, rec_params, reduction="none")
			e_sum += rec_loss.sum(-1).sum(-1).sum(-1)
			estimate = rec_loss
			estimates.append(estimate)

		# print(rec_params.max(), rec_params.min(), rec_params.mean())
		# rec_loss_avg = torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)
		rec_loss_avg = e_sum / K
		# print(rec_loss_avg.shape, kl.shape)
		# print(rec_loss_avg, kl)

		return rec_loss_avg - kl, rec_loss, kl, rec_params, rec_loss

	def generate_samples_params(self, batch, mask, K=1):
		"""
		Generate parameters of generative distributions for samples
		from the given batch.
		It makes K latent representation for each object from the batch
		and generate samples from them.
		The second axis is used to index samples for an object, i. e.
		if the batch shape is [n x D1 x D2], then the result shape is
		[n x K x D1 x D2].
		It is better to use it inside torch.no_grad in order to save memory.
		With torch.no_grad the method doesn't require extra memory
		except the memory for the result.
		"""
		_, prior = self.make_latent_distributions(batch, mask)
		samples_params = []
		for i in range(K):
			latent = prior.rsample()
			sample_params = self.generative_network(latent)
			samples_params.append(sample_params.unsqueeze(1))
		return torch.cat(samples_params, 1)


	def generate_reconstructions_params(self, batch, mask, K=1):
		"""
		Generate parameters of generative distributions for reconstructions
		from the given batch.
		It makes K latent representation for each object from the batch
		and generate samples from them.
		The second axis is used to index samples for an object, i. e.
		if the batch shape is [n x D1 x D2], then the result shape is
		[n x K x D1 x D2].
		It is better to use it inside torch.no_grad in order to save memory.
		With torch.no_grad the method doesn't require extra memory
		except the memory for the result.
		"""
		_, prior = self.make_latent_distributions(batch, mask)
		reconstructions_params = []
		for i in range(K):
			latent = prior.rsample()
			rec_params = self.generative_network(latent)
			reconstructions_params.append(rec_params.unsqueeze(1))
		return torch.cat(reconstructions_params, 1)


	def forward(self, x, middle_mask, beta, metadata):



		return self.batch_vlb(x, middle_mask=middle_mask, beta=beta, metadata=metadata)


# class VAEAC(Module):
#     """
#     Variational Autoencoder with Arbitrary Conditioning core model.
#     It is rather flexible, but have several assumptions:
#     + The batch of objects and the mask of unobserved features
#       have the same shape.
#     + The prior and proposal distributions in the latent space
#       are component-wise independent Gaussians.
#     The constructor takes
#     + Prior and proposal network which take as an input the concatenation
#       of the batch of objects and the mask of unobserved features
#       and return the parameters of Gaussians in the latent space.
#       The range of neural networks outputs should not be restricted.
#     + Generative network takes latent representation as an input
#       and returns the parameters of generative distribution
#       p_theta(x_b | z, x_{1 - b}, b), where b is the mask
#       of unobserved features. The information about x_{1 - b} and b
#       can be transmitted to generative network from prior network
#       through nn_utils.MemoryLayer. It is guaranteed that for every batch
#       prior network is always executed before generative network.
#     + Reconstruction log probability. rec_log_prob is a callable
#       which takes (groundtruth, distr_params, mask) as an input
#       and return vector of differentiable log probabilities
#       p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
#     + Sigma_mu and sigma_sigma are the coefficient of the regularization
#       in the hidden space. The default values correspond to a very weak,
#       almost disappearing regularization, which is suitable for all
#       experimental setups the model was tested on.
#     """
#     def __init__(self, rec_log_prob, proposal_network, prior_network,
#                  generative_network, mask_generator, channels=3, sigma_mu=1e4, sigma_sigma=1e-4):
#         super().__init__()
		
#         self.rec_log_prob = rec_log_prob
#         self.proposal_network = proposal_network
#         self.prior_network = prior_network
#         self.generative_network = generative_network
#         self.mask_generator = mask_generator
#         self.sigma_mu = sigma_mu
#         self.sigma_sigma = sigma_sigma
#         self.channels = channels


#     def make_observed(self, batch, mask):
#         """
#         Copy batch of objects and zero unobserved features.
#         """
#         observed = torch.tensor(batch)
#         observed[mask.bool()] = 0
#         # print(batch.sum(), observed.sum())

#         return observed


#     def make_latent_distributions(self, batch, mask, no_proposal=False):
#         """
#         Make latent distributions for the given batch and mask.
#         No no_proposal is True, return None instead of proposal distribution.
#         """
#         observed = self.make_observed(batch, mask)
#         # print(observed)
#         if no_proposal:
#             proposal = None
#         else:
#             full_info = torch.cat([batch, mask], 1)
#             proposal_params = self.proposal_network(full_info)
#             proposal = normal_parse_params(proposal_params, 1e-3)

#         prior_params = self.prior_network(torch.cat([observed, mask], 1))
#         # print(prior_params.shape, proposal_params.shape, "----------")
#         prior = normal_parse_params(prior_params, 1e-3)
#         return proposal, prior


#     def prior_regularization(self, prior):
#         """
#         The prior distribution regularization in the latent space.
#         Though it saves prior distribution parameters from going to infinity,
#         the model usually doesn't diverge even without this regularization.
#         It almost doesn't affect learning process near zero with default
#         regularization parameters which are recommended to be used.
#         """
#         num_objects = prior.mean.shape[0]
#         # mu = torch.abs(prior.mean.view(num_objects, -1)).log()
#         # sigma = prior.scale.view(num_objects, -1).log()
#         mu_og = prior.mean.view(num_objects, -1)
#         sigma_og = prior.scale.view(num_objects, -1)


#         mu_regularizer_og = -(mu_og ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
#         # mu_regularizer = - (torch.logsumexp(2 * mu, -1) - math.log(2) - 2 * math.log(self.sigma_mu)).exp()

#         sigma_regularizer_og = (sigma_og.log() - sigma_og).sum(-1) * self.sigma_sigma
#         # sigma_regularizer = ((sigma - sigma.exp()).sum(-1) + math.log(self.sigma_sigma)).exp()
#         # print(mu_regularizer, mu_regularizer_og, "----------------")
#         # print(sigma_regularizer, sigma_regularizer_og, "=====================")
#         # print("prior reg", torch.min(mu_regularizer), torch.max(mu_regularizer), torch.min(sigma_regularizer_og), torch.max(sigma_regularizer_og))
#         return mu_regularizer_og + sigma_regularizer_og

#     def batch_vlb(self, batch, mask, beta):
#         """
#         Compute differentiable lower bound for the given batch of objects
#         and mask.
#         """
#         # print(batch.shape,"====")
#         proposal, prior = self.make_latent_distributions(batch, mask)
#         prior_regularization = self.prior_regularization(prior)
#         latent = proposal.rsample()
#         # print(latent.shape, proposal.shape, prior.shape, "======")
#         rec_params = self.generative_network(latent)
#         # torchvision.utils.save_image(batch[0:5, 0].reshape(5, 1, 512, 512), "test/test_og.png", normalize=False, nrow=2)

#         # print(batch.shape, rec_params.shape, mask.shape)        
#         rec_loss = self.rec_log_prob(batch, rec_params, mask, reduction="sum")

#         kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).mean(-1)

#         # print(torch.min(rec_loss), torch.max(rec_loss), torch.min(kl), torch.max(kl))
#         return rec_loss - beta * kl + prior_regularization, rec_loss, kl


#     def features(self, batch, mask, K=1):
#         """
#         Compute differentiable lower bound for the given batch of objects
#         and mask.
#         """

#         samples_params = []

#         proposal, prior = self.make_latent_distributions(batch, mask)
#         prior_regularization = self.prior_regularization(prior)
#         # latent = proposal.rsample()
#         # latent = prior.rsample()
#         # rec_params = self.generative_network(latent)
#         # rec_loss = self.rec_log_prob(batch, rec_params, mask)
#         # rec_loss = -torch.nn.functional.l1_loss(batch * mask, rec_params[:, 0] * mask)
		
#         kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
		
#         total_rec_loss = 0

#         for i in range(K):
#             latent = prior.rsample()
#             sample_params = self.generative_network(latent)


#             total_rec_loss += self.rec_log_prob(batch, sample_params, mask, reduction="none")

#             samples_params.append(sample_params.unsqueeze(1))
			

#         return total_rec_loss, kl, prior_regularization, torch.cat(samples_params, 1)
	
#     # def get_rec(self, batch, mask):

#     #     proposal, prior = self.make_latent_distributions(batch, mask)
#     #     latent = proposal.rsample()
#     #     rec_params = self.generative_network(latent)

#     #     return rec_params, mask, proposal, prior

#     # def compute_loss(self, rec_params, mask, proposal, prior):

#     #     prior_regularization = self.prior_regularization(prior)
#     #     rec_loss = self.rec_log_prob(batch, rec_params, mask)
#     #     kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
		
#     #     return rec_loss - kl + prior_regularization


	
#     def batch_iwae(self, batch, m, K=1, middle_mask=False):
#         """
#         Compute IWAE log likelihood estimate with K samples per object.
#         Technically, it is differentiable, but it is recommended to use it
#         for evaluation purposes inside torch.no_grad in order to save memory.
#         With torch.no_grad the method almost doesn't require extra memory
#         for very large K.
#         The method makes K independent passes through generator network,
#         so the batch size is the same as for training with batch_vlb.
#         """
#         mask = torch.zeros_like(batch)
#         if middle_mask:
#             mask[:, 1] = 1
#         else:
#             for i in range(batch.shape[0]):
				
#                 # rands_xy = random.randint(16, 32) // 2
#                 rands_xy = 16

#                 rands_z = random.randint(0, 2)
				
#                 randx = random.randint(rands_xy, 192 - rands_xy)
#                 randy = random.randint(rands_xy, 192 - rands_xy)
#                 randz = random.randint(rands_z, 32 - rands_z - 1)
				

#                 mask[i, :, 
#                 # randz - rands_z: randz + rands_z + 1, 
#                 randx - rands_xy: randx + rands_xy, 
#                 randy - rands_xy: randy + rands_xy] = 1

#         # print(mask.sum(), mask.dtype)
#         # print(m.sum(), m.dtype)
#         proposal, prior = self.make_latent_distributions(batch, mask)
#         estimates = []
#         # kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
#         kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).mean(-1)

#         for i in range(K):
#             # latent = proposal.rsample()

#             # rec_params = self.generative_network(latent)
#             # rec_loss = self.rec_log_prob(batch, rec_params, mask)

#             # prior_log_prob = prior.log_prob(latent)
#             # prior_log_prob = prior_log_prob.view(batch.shape[0], -1)
#             # prior_log_prob = prior_log_prob.sum(-1)

#             # proposal_log_prob = proposal.log_prob(latent)
#             # proposal_log_prob = proposal_log_prob.view(batch.shape[0], -1)
#             # proposal_log_prob = proposal_log_prob.sum(-1)

#             # estimate = rec_loss + prior_log_prob - proposal_log_prob

#             latent = prior.rsample()
#             rec_params = self.generative_network(latent)
#             rec_loss = self.rec_log_prob(batch, rec_params, mask, reduction="none")

#             estimate = rec_loss.sum()
#             estimates.append(estimate[:, None])

#         # print(rec_params.max(), rec_params.min(), rec_params.mean())
#         rec_loss = torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)
#         return rec_loss - kl, rec_loss, kl, rec_params, mask

#     def generate_samples_params(self, batch, mask, K=1):
#         """
#         Generate parameters of generative distributions for samples
#         from the given batch.
#         It makes K latent representation for each object from the batch
#         and generate samples from them.
#         The second axis is used to index samples for an object, i. e.
#         if the batch shape is [n x D1 x D2], then the result shape is
#         [n x K x D1 x D2].
#         It is better to use it inside torch.no_grad in order to save memory.
#         With torch.no_grad the method doesn't require extra memory
#         except the memory for the result.
#         """
#         _, prior = self.make_latent_distributions(batch, mask)
#         samples_params = []
#         for i in range(K):
#             latent = prior.rsample()
#             sample_params = self.generative_network(latent)
#             samples_params.append(sample_params.unsqueeze(1))
#         return torch.cat(samples_params, 1)

#     def generate_reconstructions_params(self, batch, mask, K=1):
#         """
#         Generate parameters of generative distributions for reconstructions
#         from the given batch.
#         It makes K latent representation for each object from the batch
#         and generate samples from them.
#         The second axis is used to index samples for an object, i. e.
#         if the batch shape is [n x D1 x D2], then the result shape is
#         [n x K x D1 x D2].
#         It is better to use it inside torch.no_grad in order to save memory.
#         With torch.no_grad the method doesn't require extra memory
#         except the memory for the result.
#         """
#         _, prior = self.make_latent_distributions(batch, mask)
#         reconstructions_params = []
#         for i in range(K):
#             latent = prior.rsample()
#             rec_params = self.generative_network(latent)
#             reconstructions_params.append(rec_params.unsqueeze(1))
#         return torch.cat(reconstructions_params, 1)

#     def forward(self, x, middle_mask=False, beta=1):
		
#         mask = torch.zeros_like(x)
		
#         if middle_mask:
#             mask[:, 1] = 1

#         else:
#             for i in range(x.shape[0]):
				
#                 # rands_xy = random.randint(16, 32) // 2
#                 rands_xy = 16
#                 # rands_z = random.randint(0, 2)
				
#                 randx = random.randint(rands_xy, 192 - rands_xy)
#                 randy = random.randint(rands_xy, 192 - rands_xy)
#                 # randz = random.randint(rands_z, 32 - rands_z - 1)
				

#                 mask[i, :, 
#                 # randz - rands_z: randz + rands_z + 1, 
#                 randx - rands_xy: randx + rands_xy, 
#                 randy - rands_xy: randy + rands_xy] = 1

#                 # mask[i, 1:-1, 
#                 # # randz - rands_z: randz + rands_z + 1, 
#                 # randx - rands_xy: randx + rands_xy, 
#                 # randy - rands_xy: randy + rands_xy] = 1


#         return self.batch_vlb(x, mask, beta)


# class CVAE(Module):
#     """
#     Variational Autoencoder with Arbitrary Conditioning core model.
#     It is rather flexible, but have several assumptions:
#     + The batch of objects and the mask of unobserved features
#       have the same shape.
#     + The prior and proposal distributions in the latent space
#       are component-wise independent Gaussians.
#     The constructor takes
#     + Prior and proposal network which take as an input the concatenation
#       of the batch of objects and the mask of unobserved features
#       and return the parameters of Gaussians in the latent space.
#       The range of neural networks outputs should not be restricted.
#     + Generative network takes latent representation as an input
#       and returns the parameters of generative distribution
#       p_theta(x_b | z, x_{1 - b}, b), where b is the mask
#       of unobserved features. The information about x_{1 - b} and b
#       can be transmitted to generative network from prior network
#       through nn_utils.MemoryLayer. It is guaranteed that for every batch
#       prior network is always executed before generative network.
#     + Reconstruction log probability. rec_log_prob is a callable
#       which takes (groundtruth, distr_params, mask) as an input
#       and return vector of differentiable log probabilities
#       p_theta(x_b | z, x_{1 - b}, b) for each object of the batch.
#     + Sigma_mu and sigma_sigma are the coefficient of the regularization
#       in the hidden space. The default values correspond to a very weak,
#       almost disappearing regularization, which is suitable for all
#       experimental setups the model was tested on.
#     """
#     def __init__(self, rec_log_prob, proposal_network, prior_network,
#                  generative_network, mask_generator, channels=3, sigma_mu=1e4, sigma_sigma=1e-4):
#         super().__init__()
		
#         self.rec_log_prob = rec_log_prob
#         self.proposal_network = proposal_network
#         self.prior_network = prior_network
#         self.generative_network = generative_network

#         self.mask_generator = mask_generator
#         self.sigma_mu = sigma_mu
#         self.sigma_sigma = sigma_sigma
#         self.channels = channels


#     def make_observed(self, batch, mask):
#         """
#         Copy batch of objects and zero unobserved features.
#         """
#         observed = torch.tensor(batch)
#         observed[observed > 0.6] = 0.6
#         # print(batch.sum(), observed.sum())

#         return observed


#     def make_latent_distributions(self, batch, mask, no_proposal=False):
#         """
#         Make latent distributions for the given batch and mask.
#         No no_proposal is True, return None instead of proposal distribution.
#         """
#         observed = self.make_observed(batch, mask)
#         # print(observed)
#         if no_proposal:
#             proposal = None
#         else:
#             # full_info = torch.cat([batch, mask], 1)
#             full_info = batch

#             proposal_params = self.proposal_network(full_info)
#             proposal = normal_parse_params(proposal_params, 1e-3)

#         # prior_params = self.prior_network(torch.cat([observed, mask], 1))
#         prior_params = self.prior_network(observed)

#         # print(prior_params.shape, proposal_params.shape, "----------")
#         prior = normal_parse_params(prior_params, 1e-3)
#         return proposal, prior


#     def prior_regularization(self, prior):
#         """
#         The prior distribution regularization in the latent space.
#         Though it saves prior distribution parameters from going to infinity,
#         the model usually doesn't diverge even without this regularization.
#         It almost doesn't affect learning process near zero with default
#         regularization parameters which are recommended to be used.
#         """
#         num_objects = prior.mean.shape[0]
#         # mu = torch.abs(prior.mean.view(num_objects, -1)).log()
#         # sigma = prior.scale.view(num_objects, -1).log()
#         mu_og = prior.mean.view(num_objects, -1)
#         sigma_og = prior.scale.view(num_objects, -1)


#         mu_regularizer_og = -(mu_og ** 2).sum(-1) / 2 / (self.sigma_mu ** 2)
#         # mu_regularizer = - (torch.logsumexp(2 * mu, -1) - math.log(2) - 2 * math.log(self.sigma_mu)).exp()

#         sigma_regularizer_og = (sigma_og.log() - sigma_og).sum(-1) * self.sigma_sigma
#         # sigma_regularizer = ((sigma - sigma.exp()).sum(-1) + math.log(self.sigma_sigma)).exp()
#         # print(mu_regularizer, mu_regularizer_og, "----------------")
#         # print(sigma_regularizer, sigma_regularizer_og, "=====================")
#         # print("prior reg", torch.min(mu_regularizer), torch.max(mu_regularizer), torch.min(sigma_regularizer_og), torch.max(sigma_regularizer_og))
#         return mu_regularizer_og + sigma_regularizer_og

#     def batch_vlb(self, batch, mask, beta):
#         """
#         Compute differentiable lower bound for the given batch of objects
#         and mask.
#         """
#         # print(batch.shape,"====")
#         proposal, prior = self.make_latent_distributions(batch, mask)
#         prior_regularization = self.prior_regularization(prior)
#         latent = proposal.rsample()
#         # print(latent.shape, proposal.shape, prior.shape, "======")
#         rec_params = self.generative_network(latent)
#         # torchvision.utils.save_image(batch[0:5, 0].reshape(5, 1, 512, 512), "test/test_og.png", normalize=False, nrow=2)

#         # print(batch.shape, rec_params.shape, mask.shape)        
#         rec_loss = self.rec_log_prob(batch, rec_params, mask, reduction="mean")

#         kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).mean(-1)

#         # print(torch.min(rec_loss), torch.max(rec_loss), torch.min(kl), torch.max(kl))
#         return rec_loss - beta * kl + prior_regularization, rec_loss, kl


#     def features(self, batch, mask, K=1):
#         """
#         Compute differentiable lower bound for the given batch of objects
#         and mask.
#         """

#         samples_params = []

#         proposal, prior = self.make_latent_distributions(batch, mask)
#         prior_regularization = self.prior_regularization(prior)
#         # latent = proposal.rsample()
#         # latent = prior.rsample()
#         # rec_params = self.generative_network(latent)
#         # rec_loss = self.rec_log_prob(batch, rec_params, mask)
#         # rec_loss = -torch.nn.functional.l1_loss(batch * mask, rec_params[:, 0] * mask)
		
#         kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
		
#         total_rec_loss = 0

#         for i in range(K):
#             latent = prior.rsample()
#             sample_params = self.generative_network(latent)


#             total_rec_loss += self.rec_log_prob(batch, sample_params, mask, reduction="none")

#             samples_params.append(sample_params.unsqueeze(1))
			

#         return total_rec_loss, kl, prior_regularization, torch.cat(samples_params, 1)
	
#     # def get_rec(self, batch, mask):

#     #     proposal, prior = self.make_latent_distributions(batch, mask)
#     #     latent = proposal.rsample()
#     #     rec_params = self.generative_network(latent)

#     #     return rec_params, mask, proposal, prior

#     # def compute_loss(self, rec_params, mask, proposal, prior):

#     #     prior_regularization = self.prior_regularization(prior)
#     #     rec_loss = self.rec_log_prob(batch, rec_params, mask)
#     #     kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
		
#     #     return rec_loss - kl + prior_regularization


	
#     def batch_iwae(self, batch, m, K, middle_mask=False):
#         """
#         Compute IWAE log likelihood estimate with K samples per object.
#         Technically, it is differentiable, but it is recommended to use it
#         for evaluation purposes inside torch.no_grad in order to save memory.
#         With torch.no_grad the method almost doesn't require extra memory
#         for very large K.
#         The method makes K independent passes through generator network,
#         so the batch size is the same as for training with batch_vlb.
#         """

#         mask = torch.zeros_like(batch)
		

#         # print(mask.sum(), mask.dtype)
#         # print(m.sum(), m.dtype)
#         proposal, prior = self.make_latent_distributions(batch, mask)
#         estimates = []
#         # kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).sum(-1)
#         kl = kl_divergence(proposal, prior).view(batch.shape[0], -1).mean(-1)

#         for i in range(K):
#             # latent = proposal.rsample()

#             # rec_params = self.generative_network(latent)
#             # rec_loss = self.rec_log_prob(batch, rec_params, mask)

#             # prior_log_prob = prior.log_prob(latent)
#             # prior_log_prob = prior_log_prob.view(batch.shape[0], -1)
#             # prior_log_prob = prior_log_prob.sum(-1)

#             # proposal_log_prob = proposal.log_prob(latent)
#             # proposal_log_prob = proposal_log_prob.view(batch.shape[0], -1)
#             # proposal_log_prob = proposal_log_prob.sum(-1)

#             # estimate = rec_loss + prior_log_prob - proposal_log_prob

#             latent = prior.rsample()
#             rec_params = self.generative_network(latent)
#             rec_loss = self.rec_log_prob(batch, rec_params, mask, reduction="mean")


#             estimate = rec_loss
#             estimates.append(estimate[:, None])

#         # print(rec_params.max(), rec_params.min(), rec_params.mean())
#         rec_loss = torch.logsumexp(torch.cat(estimates, 1), 1) - math.log(K)
#         return rec_loss - kl, rec_loss, kl, rec_params, mask

#     def generate_samples_params(self, batch, mask, K=1):
#         """
#         Generate parameters of generative distributions for samples
#         from the given batch.
#         It makes K latent representation for each object from the batch
#         and generate samples from them.
#         The second axis is used to index samples for an object, i. e.
#         if the batch shape is [n x D1 x D2], then the result shape is
#         [n x K x D1 x D2].
#         It is better to use it inside torch.no_grad in order to save memory.
#         With torch.no_grad the method doesn't require extra memory
#         except the memory for the result.
#         """
#         _, prior = self.make_latent_distributions(batch, mask)
#         samples_params = []
#         for i in range(K):
#             latent = prior.rsample()
#             sample_params = self.generative_network(latent)
#             samples_params.append(sample_params.unsqueeze(1))
#         return torch.cat(samples_params, 1)

#     def generate_reconstructions_params(self, batch, mask, K=1):
#         """
#         Generate parameters of generative distributions for reconstructions
#         from the given batch.
#         It makes K latent representation for each object from the batch
#         and generate samples from them.
#         The second axis is used to index samples for an object, i. e.
#         if the batch shape is [n x D1 x D2], then the result shape is
#         [n x K x D1 x D2].
#         It is better to use it inside torch.no_grad in order to save memory.
#         With torch.no_grad the method doesn't require extra memory
#         except the memory for the result.
#         """
#         _, prior = self.make_latent_distributions(batch, mask)
#         reconstructions_params = []
#         for i in range(K):
#             latent = prior.rsample()
#             rec_params = self.generative_network(latent)
#             reconstructions_params.append(rec_params.unsqueeze(1))
#         return torch.cat(reconstructions_params, 1)

#     def forward(self, x, middle_mask=False, beta=1):
		
#         mask = torch.zeros_like(x)
		
#         if middle_mask:
#             mask[:, 1] = 1

#         else:
#             for i in range(x.shape[0]):
				
#                 # rands_xy = random.randint(16, 32) // 2
#                 rands_xy = 16
#                 # rands_z = random.randint(0, 2)
				
#                 randx = random.randint(rands_xy, 192 - rands_xy)
#                 randy = random.randint(rands_xy, 192 - rands_xy)
#                 # randz = random.randint(rands_z, 32 - rands_z - 1)
				

#                 mask[i, :, 
#                 # randz - rands_z: randz + rands_z + 1, 
#                 randx - rands_xy: randx + rands_xy, 
#                 randy - rands_xy: randy + rands_xy] = 1

#                 # mask[i, 1:-1, 
#                 # # randz - rands_z: randz + rands_z + 1, 
#                 # randx - rands_xy: randx + rands_xy, 
#                 # randy - rands_xy: randy + rands_xy] = 1


#         return self.batch_vlb(x, mask, beta)

