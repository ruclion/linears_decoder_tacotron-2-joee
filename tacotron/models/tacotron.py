import tensorflow as tf
from infolog import log
from tacotron.models.Architecture_wrappers import TacotronDecoderCell, TacotronEncoderCell
from tacotron.models.attention import LocationSensitiveAttention
from tacotron.models.custom_decoder import CustomDecoder
from tacotron.models.helpers import TacoTestHelper, TacoTrainingHelper
from tacotron.models.modules import *
from tacotron.utils.symbols import symbols
from tensorflow.contrib.seq2seq import dynamic_decode
from tacotron.models.flipGradientTF import GradientReversal
from tacotron.models.flip_gradient import flip_gradient


class Tacotron():
	"""Tacotron-2 Feature prediction Model.
	"""
	def __init__(self, hparams):
		self._hparams = hparams


	def initialize(self, inputs, input_speaker_id, input_lengths, mel_targets=None, stop_token_targets=None, linear_targets=None, targets_lengths=None, gta=False,
			global_step=None, is_training=False, is_evaluating=False):
		"""
		Initializes the model for inference

		sets "mel_outputs" and "alignments" fields.

		Args:
			- inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
			  steps in the input time series, and values are character IDs
			- input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
			of each sequence in inputs.
			- mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
			of steps in the output time series, M is num_mels, and values are entries in the mel
			spectrogram. Only needed for training.
		"""
		if mel_targets is None and stop_token_targets is not None:
			raise ValueError('no mel targets were provided but token_targets were given')
		if mel_targets is not None and stop_token_targets is None and not gta:
			raise ValueError('Mel targets are provided without corresponding token_targets')
		if not gta and self._hparams.predict_linear==True and linear_targets is None and is_training:
			raise ValueError('Model is set to use post processing to predict linear spectrograms in training but no linear targets given!')
		if gta and linear_targets is not None:
			raise ValueError('Linear spectrogram prediction is not supported in GTA mode!')
		if is_training and self._hparams.mask_decoder and targets_lengths is None:
			raise RuntimeError('Model set to mask paddings but no targets lengths provided for the mask!')
		if is_training and is_evaluating:
			raise RuntimeError('Model can not be in training and evaluation modes at the same time!')

		with tf.variable_scope('inference') as scope:
			batch_size = tf.shape(inputs)[0]
			hp = self._hparams
			assert hp.tacotron_teacher_forcing_mode in ('constant', 'scheduled')
			if hp.tacotron_teacher_forcing_mode == 'scheduled' and is_training:
				assert global_step is not None

			#GTA is only used for predicting mels to train Wavenet vocoder, so we ommit post processing when doing GTA synthesis
			post_condition = hp.predict_linear and not gta

			# Embeddings ==> [batch_size, sequence_length, embedding_dim]
			embedding_table = tf.get_variable(
				'inputs_embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32)
			embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)
			
			# Speaker Embeddings ==> [batch_size, embedding_dim]
			self.speaker_id_embedding_table = tf.get_variable('input_speaker_id_embedding', [hp.speaker_num, hp.speaker_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.5))
			embedded_speaker_id = tf.nn.embedding_lookup(self.speaker_id_embedding_table, input_speaker_id)


			#Encoder Cell ==> [batch_size, encoder_steps, encoder_lstm_units]
			encoder_cell = TacotronEncoderCell(
				EncoderConvolutions(is_training, hparams=hp, scope='encoder_convolutions'),
				EncoderRNN(is_training, size=hp.encoder_lstm_units,
					zoneout=hp.tacotron_zoneout_rate, scope='encoder_LSTM'))

			encoder_outputs = encoder_cell(embedded_inputs, input_lengths)
			
			#first change encoder_outputs to concated version.
			
			#second add. need same dims 
			#encoder_outputs = encoder_outputs + embedded_speaker_id
			'''
			#first concat.
			input_seq_len = tf.shape(encoder_outputs)[1]
			print('!!!!!!!!!!before tile')
			embedded_speaker_id = tf.expand_dims(embedded_speaker_id, 1)
			embedded_speaker_id = tf.tile(embedded_speaker_id, multiples=[1, input_seq_len, 1])
			print('!!!!!!!!!!after tile')
			id_encoder_outputs = tf.concat([encoder_outputs, embedded_speaker_id], axis=-1)
			'''
			id_encoder_outputs = encoder_outputs
			#still use encoder_outputs
			
			#use keras version, but not run.
			'''
			print('hhhhhhhhhhhhhhhhhhhhhhhhhhhh')
			hp_lambda = 1.0
			Flip = GradientReversal(hp_lambda)
			Flip_encoder_outputs = Flip(encoder_outputs)
			'''
			'''
			#use tensorflow version, but star's is only 5 and i don't understand.
			Flip_encoder_outputs = flip_gradient(encoder_outputs, l=1.0)
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', Flip_encoder_outputs, type(Flip_encoder_outputs))
			densed_256_encoder_outputs = tf.layers.dense(Flip_encoder_outputs, 256, tf.nn.relu)
			softmax_encoder_outputs = tf.layers.dense(densed_256_encoder_outputs, hp.speaker_num, tf.nn.softmax)
			
			long_speaker_id = tf.reshape(input_speaker_id, shape = [tf.shape(inputs)[0], 1])
			tiled_speaker_id = tf.tile(long_speaker_id, multiples=[1, tf.shape(softmax_encoder_outputs)[1]])
			print('tiled_speaker_id', tiled_speaker_id)
			one_hot_speaker_id = tf.one_hot(tiled_speaker_id, depth=hp.speaker_num)
			print('one_hot_speaker_id', one_hot_speaker_id)
			#self.one_hot_speaker_id and self.softmax_encoder_outputs is at below
			#long_speaker_id = tf.expand_dims(long_speaker_id, axis=2)
			#dann_out = Dense(2)(dann_in)
			#Flip_encoder_outputs = 
			'''
			#For shape visualization purpose
			enc_conv_output_shape = encoder_cell.conv_output_shape


			#Decoder Parts
			#Attention Decoder Prenet
			prenet = Prenet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate, scope='decoder_prenet')
			#Attention Mechanism
			attention_mechanism = LocationSensitiveAttention(hp.attention_dim, id_encoder_outputs, hparams=hp,
				is_training=is_training, mask_encoder=hp.mask_encoder, memory_sequence_length=input_lengths,
				smoothing=hp.smoothing, cumulate_weights=hp.cumulative_weights)
			#Decoder LSTM Cells
			decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers,
				size=hp.decoder_lstm_units, zoneout=hp.tacotron_zoneout_rate, scope='decoder_lstm')
			#Frames Projection layer
			frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, scope='linear_transform')
			#<stop_token> projection layer
			stop_projection = StopProjection(is_training or is_evaluating, shape=hp.outputs_per_step, scope='stop_token_projection')


			#Decoder Cell ==> [batch_size, decoder_steps, num_mels * r] (after decoding)
			decoder_cell = TacotronDecoderCell(
				prenet,
				attention_mechanism,
				decoder_lstm,
				frame_projection,
				stop_projection)


			#Define the helper for our decoder
			if is_training or is_evaluating or gta:
				self.helper = TacoTrainingHelper(batch_size, mel_targets, hp, gta, is_evaluating, global_step)
			else:
				self.helper = TacoTestHelper(batch_size, hp)


			#initial decoder state
			decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

			#Only use max iterations at synthesis time
			max_iters = hp.max_iters if not (is_training or is_evaluating) else None

			#Decode
			(frames_prediction, stop_token_prediction, _), final_decoder_state, _ = dynamic_decode(
				CustomDecoder(decoder_cell, self.helper, decoder_init_state),
				impute_finished=False,
				maximum_iterations=max_iters,
				swap_memory=hp.tacotron_swap_with_cpu)


			# Reshape outputs to be one output per entry
			#==> [batch_size, non_reduced_decoder_steps (decoder_steps * r), num_mels]
			decoder_output = tf.reshape(frames_prediction, [batch_size, -1, hp.num_mels])
			stop_token_prediction = tf.reshape(stop_token_prediction, [batch_size, -1])


			#Postnet
			postnet = Postnet(is_training, hparams=hp, scope='postnet_convolutions')

			#Compute residual using post-net ==> [batch_size, decoder_steps * r, postnet_channels]
			residual = postnet(decoder_output)

			#Project residual to same dimension as mel spectrogram
			#==> [batch_size, decoder_steps * r, num_mels]
			residual_projection = FrameProjection(hp.num_mels, scope='postnet_projection')
			projected_residual = residual_projection(residual)


			#Compute the mel spectrogram
			mel_outputs = decoder_output + projected_residual


			if post_condition:
				# Add post-processing CBHG. This does a great job at extracting features from mels before projection to Linear specs.
				post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size, [hp.cbhg_projection, hp.num_mels],
					hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers,
					hp.cbhg_highway_units, hp.cbhg_rnn_units, hp.batch_norm_position, is_training, name='CBHG_postnet')

				#[batch_size, decoder_steps(mel_frames), cbhg_channels]
				post_outputs = post_cbhg(mel_outputs, None)

				#Linear projection of extracted features to make linear spectrogram
				linear_specs_projection = FrameProjection(hp.num_freq, scope='cbhg_linear_specs_projection')

				#[batch_size, decoder_steps(linear_frames), num_freq]
				linear_outputs = linear_specs_projection(post_outputs)


			#Grab alignments from the final decoder state
			alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

			if is_training:
				self.ratio = self.helper._ratio
			self.inputs = inputs
			self.input_speaker_id = input_speaker_id
			#self.one_hot_speaker_id and self.softmax_encoder_outputs
			#self.softmax_encoder_outputs = softmax_encoder_outputs
			#self.one_hot_speaker_id = one_hot_speaker_id
			self.input_lengths = input_lengths
			self.decoder_output = decoder_output
			self.alignments = alignments
			self.stop_token_prediction = stop_token_prediction
			self.stop_token_targets = stop_token_targets
			self.mel_outputs = mel_outputs
			if post_condition:
				self.linear_outputs = linear_outputs
				self.linear_targets = linear_targets
			self.mel_targets = mel_targets
			self.targets_lengths = targets_lengths
			log('Initialized Tacotron model. Dimensions (? = dynamic shape): ')
			log('  Train mode:               {}'.format(is_training))
			log('  Eval mode:                {}'.format(is_evaluating))
			log('  GTA mode:                 {}'.format(gta))
			log('  Synthesis mode:           {}'.format(not (is_training or is_evaluating)))
			log('  embedding:                {}'.format(embedded_inputs.shape))
			log('  enc conv out:             {}'.format(enc_conv_output_shape))
			log('  encoder out:              {}'.format(encoder_outputs.shape))
			log('  id encoder out:              {}'.format(id_encoder_outputs.shape))
			log('  decoder out:              {}'.format(decoder_output.shape))
			log('  residual out:             {}'.format(residual.shape))
			log('  projected residual out:   {}'.format(projected_residual.shape))
			log('  mel out:                  {}'.format(mel_outputs.shape))
			if post_condition:
				log('  linear out:               {}'.format(linear_outputs.shape))
			log('  <stop_token> out:         {}'.format(stop_token_prediction.shape))


	def add_loss(self):
		'''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
		with tf.variable_scope('loss') as scope:
			hp = self._hparams

			if hp.mask_decoder:
				# Compute loss of predictions before postnet
				before = MaskedMSE(self.mel_targets, self.decoder_output, self.targets_lengths,
					hparams=self._hparams)
				# Compute loss after postnet
				after = MaskedMSE(self.mel_targets, self.mel_outputs, self.targets_lengths,
					hparams=self._hparams)
				#Compute <stop_token> loss (for learning dynamic generation stop)
				stop_token_loss = MaskedSigmoidCrossEntropy(self.stop_token_targets,
					self.stop_token_prediction, self.targets_lengths, hparams=self._hparams)
				
				#Compute <speaker_id> loss (for learning dynamic generation stop)
				#speaker_id_loss = tf.abs(-tf.reduce_mean(self.one_hot_speaker_id * tf.log(self.softmax_encoder_outputs)))
				
				#Compute masked linear loss
				if hp.predict_linear:
					#Compute Linear L1 mask loss (priority to low frequencies)
					linear_loss = MaskedLinearLoss(self.linear_targets, self.linear_outputs,
						self.targets_lengths, hparams=self._hparams)
				else:
					linear_loss=0.
			else:
				# guided_attention loss
				N = self._hparams.max_text_length
				T = self._hparams.max_mel_frames // self._hparams.outputs_per_step
				A = tf.pad(self.alignments, [(0, 0), (0, N), (0, T)], mode="CONSTANT", constant_values=-1.)[:, :N, :T]
				gts = tf.convert_to_tensor(GuidedAttention(N, T))
				attention_masks = tf.to_float(tf.not_equal(A, -1))
				attention_loss = tf.reduce_sum(tf.abs(A * gts) * attention_masks)
				attention_loss /= tf.reduce_sum(attention_masks)
				# Compute loss of predictions before postnet
				before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_output)
				# Compute loss after postnet
				after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
				#Compute <stop_token> loss (for learning dynamic generation stop)
				stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
					labels=self.stop_token_targets,
					logits=self.stop_token_prediction))
				#Compute <speaker_id> loss (for learning dynamic generation stop)
				#speaker_id_loss = tf.abs(-tf.reduce_mean(self.one_hot_speaker_id * tf.log(self.softmax_encoder_outputs)))

				if hp.predict_linear:
					#Compute linear loss
					#From https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py
					#Prioritize loss for frequencies under 2000 Hz.
					# l1 = tf.abs(self.linear_targets - self.linear_outputs)
					# n_priority_freq = int(4000 / (hp.sample_rate * 0.5) * hp.num_freq)
					# linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:,:,0:n_priority_freq])
					linear_loss = tf.losses.mean_squared_error(self.linear_targets, self.linear_outputs)
				else:
					linear_loss = 0.

			# Compute the regularization weight
			if hp.tacotron_scale_regularization:
				reg_weight_scaler = 1. / (2 * hp.max_abs_value) if hp.symmetric_mels else 1. / (hp.max_abs_value)
				reg_weight = hp.tacotron_reg_weight * reg_weight_scaler
			else:
				reg_weight = hp.tacotron_reg_weight

			# Get all trainable variables
			all_vars = tf.trainable_variables()
			regularization = tf.add_n([tf.nn.l2_loss(v) for v in all_vars
				if not('bias' in v.name or 'Bias' in v.name)]) * reg_weight

			# Compute final loss term
			self.before_loss = before
			self.after_loss = after
			self.stop_token_loss = stop_token_loss
			#speaker_id_loss
			#self.speaker_id_loss = 0.002 * speaker_id_loss #Paper said: 0.002
			
			self.regularization_loss = regularization
			self.linear_loss = linear_loss
			self.attention_loss = attention_loss
			self.loss = self.before_loss + self.after_loss + self.stop_token_loss + self.regularization_loss + self.linear_loss + self.attention_loss
			#self.loss = self.before_loss + self.after_loss + self.stop_token_loss + self.speaker_id_loss + self.regularization_loss + self.linear_loss + self.attention_loss

	def add_optimizer(self, global_step):
		'''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

		Args:
			global_step: int32 scalar Tensor representing current global step in training
		'''
		with tf.variable_scope('optimizer') as scope:
			hp = self._hparams
			if hp.tacotron_decay_learning_rate:
				self.decay_steps = hp.tacotron_decay_steps
				self.decay_rate = hp.tacotron_decay_rate
				self.learning_rate = self._learning_rate_decay(hp.tacotron_initial_learning_rate, global_step)
			else:
				self.learning_rate = tf.convert_to_tensor(hp.tacotron_initial_learning_rate)

			optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.tacotron_adam_beta1,
				hp.tacotron_adam_beta2, hp.tacotron_adam_epsilon)
			gradients, variables = zip(*optimizer.compute_gradients(self.loss))
			self.gradients = gradients
			#0.5 clip for Flip_encoder_outputs, but i don't want to write special, use below
			#sinclipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
			
			#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			#gvs = optimizer.compute_gradients(cost)
			#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
			#gradients, variables = gvs
			
			#train_op = optimizer.apply_gradients(capped_gvs)
			
			#Just for causion
			#https://github.com/Rayhane-mamah/Tacotron-2/issues/11
			if hp.tacotron_clip_gradients:
				clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.)
			else:
				clipped_gradients = gradients

			# Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
			# https://github.com/tensorflow/tensorflow/issues/1122
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
					global_step=global_step)

	def _learning_rate_decay(self, init_lr, global_step):
		#################################################################
		# Narrow Exponential Decay:

		# Phase 1: lr = 1e-3
		# We only start learning rate decay after 50k steps

		# Phase 2: lr in ]1e-5, 1e-3[
		# decay reach minimal value at step 310k

		# Phase 3: lr = 1e-5
		# clip by minimal learning rate value (step > 310k)
		#################################################################
		hp = self._hparams

		#Compute natural exponential decay
		lr = tf.train.exponential_decay(init_lr,
			global_step - hp.tacotron_start_decay, #lr = 1e-3 at step 50k
			self.decay_steps,
			self.decay_rate, #lr = 1e-5 around step 310k
			name='lr_exponential_decay')


		#clip learning rate by max and min values (initial and final values)
		return tf.minimum(tf.maximum(lr, hp.tacotron_final_learning_rate), init_lr)
