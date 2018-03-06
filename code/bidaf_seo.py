# h: [N, M, JX, 2d]
# u: [N, JQ, 2d] 
# 
# h_mask: [N, M, JX]
# u_mask: [N, JQ]
#
# N = batch size
# M = max number of context sentences
# JX = max context sentence size
# JQ = max question size
# --> "num_context" = M*JX
# --> "num_question" = JQ
# --> "context_vec_size" = 2d


#######################################################
## Understanding original BiDAF coode
#######################################################

# JX = tf.shape(h)[2]
# M  = tf.shape(h)[1]
# JQ = tf.shape(u)[1]

# ## RECONFIG INPUT
# h_exp = tf.expand_dims(h, 3) # (N, M, JX, 1, 2d)
# u_exp = tf.expand_dims(u, 1) # (N, 1, JQ, 2d)
# u_exp = tf.expand_dims(u, 1) # (N, 1, 1, JQ, 2d)

# h_aug = tf.tile(h_exp, [1, 1, 1, JQ, 1]) # (N, M, JX, JQ, 2d)
# u_aug = tf.tile(u_exp, [1, M, JX, 1, 1]) # (N, M, JX, JQ, 2d)

# ## RECONFIG MASKS
# h_mask_exp = tf.expand_dims(h_mask, 3)      # (N, M, JX, 1)
# u_mask_exp = tf.expand_dims(u_mask, 1)      # (N, 1, JQ)
# u_mask_exp = tf.expand_dims(u_mask_exp, 1)  # (N, 1, 1, JQ)

# h_mask_aug = tf.tile(h_mask_exp, [1, 1, 1, JQ]) # (N, M, JX, JQ)
# u_mask_aug = tf.tile(u_mask_exp, [1, M, JX, 1]) # (M, M, JX, JQ)

# logits_input = [h_aug, u_aug, h_aug*u_aug]
# logits_mask  = h_mask_aug & u_mask_aug

# logits = linear(logits_input, )
# logits = exp_mask(logits, logits_mask)

# output = tf.concat(3, [h, u_a, h * u_a, h * h_a])
# output = tf.nn.dropout(output, self.keep_prob)


#######################################################
## Using project notation
#######################################################

# ## RECONFIG INPUT
# h_exp = tf.expand_dims(h, 2) # (N, Nc, 1, 2d)
# u_exp = tf.expand_dims(u, 1) # (N, 1, Nq, 2d)

# h_aug = tf.tile(h_exp, [1, 1, Nq, 1]) # (N, Nc, Nq, 2d)
# u_aug = tf.tile(u_exp, [1, Nc, 1, 1]) # (N, Nc, Nq, 2d)

# ## RECONFIG MASKS
# h_mask_exp = tf.expand_dims(h_mask, 2)      # (N, Nc, 1)
# u_mask_exp = tf.expand_dims(u_mask, 1)      # (N, 1, Nq)

# h_mask_aug = tf.tile(h_mask_exp, [1, 1, Nq]) # (N, Nc, Nq)
# u_mask_aug = tf.tile(u_mask_exp, [1, Nc, 1]) # (N, Nc, Nq)
# hu_mask    = h_mask_aug & u_mask_aug
