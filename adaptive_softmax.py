import tensorflow as tf


class AdaptiveSoftmax(tf.keras.layers.Layer):
    def __init__(self, input_dim, cutoff, project_factor=4, project_dims=None, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.cluster_num = len(cutoff) - 1
        if project_dims:
            assert (len(project_dims) == self.cluster_num)
        else:
            project_dims = []
            tail_project_factor = project_factor
            for i in range(self.cluster_num):
                dim = max(1, input_dim / tail_project_factor)
                project_dims.append(dim)
                tail_project_factor *= project_factor

        self.cutoff = cutoff
        head_dim = cutoff[0] + self.cluster_num
        self.head_proj = self.add_weight(name="adaptive_softmax_head_proj_w", shape=[input_dim, input_dim])
        self.head_w = self.add_weight(name="adaptive_softmax_head_w", shape=[input_dim, head_dim])
        self.head_b = self.add_weight(name="adaptive_softmax_head_b", shape=[head_dim],
                                      initializer=tf.constant_initializer(0.0, dtype=tf.float32))

        self.tail_w = []
        for i in range(self.cluster_num):
            project_dim = project_dims[i]
            tail_dim = cutoff[i + 1] - cutoff[i]
            self.tail_w.append([
                self.add_weight(name="adaptive_softmax_tail{}_proj_w".format(i + 1), shape=[input_dim, project_dim]),
                self.add_weight(name="adaptive_softmax_tail{}_w".format(i + 1), shape=[project_dim, tail_dim]),
                self.add_weight(name="adaptive_softmax_tail{}_b".format(i + 1), shape=[tail_dim],
                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            ])

    def call(self, inputs, labels, loss_mode='likelihood', mode='train', **kwargs):
        """
        Args:
        inputs: float Tensor.
        labels: a list of int32 or int64 ids
        loss_mode: either 'likelihood' or 'likelihood+' (which is the sum over p_i log(p_i))
        mode:'train' or 'inference'
        Returns:
        loss and probablity vector
        """
        loss = self.loss(inputs, labels)
        softmax = None
        if loss_mode == "likelihood+":
            softmax = self.softmax(inputs)
            loss += tf.reduce_sum(tf.multiply(softmax, tf.math.log(softmax)))
        if mode == 'inference':
            if loss_mode == "likelihood+":
                softmax = self.softmax(inputs)
        return loss, softmax

    def loss(self, inputs, labels, name='loss'):
        # Get tail masks and update head labels
        head_labels = labels
        ones = tf.ones([tf.size(labels)], dtype=tf.int32)
        for i in range(self.cluster_num):
            mask = tf.logical_and(tf.greater_equal(labels, self.cutoff[i]), tf.less(labels, self.cutoff[i + 1]))

            # Update head labels
            head_labels = tf.where(mask, ones * (self.cutoff[0] + i), head_labels)

            # Compute tail loss
            tail_inputs = tf.boolean_mask(inputs, mask)
            tail_logits = tf.matmul(self.gelu(tf.matmul(tail_inputs, self.tail_w[i][0])), self.tail_w[i][1]) + \
                          self.tail_w[i][2]
            tail_labels = tf.boolean_mask(labels - self.cutoff[i], mask)
            tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_labels)
            aligned_tail_loss = tf.SparseTensor(tf.squeeze(tf.where(mask)), tail_loss,
                                                [tf.size(labels, out_type=tf.int64)])
            loss = tf.sparse_tensor_to_dense(aligned_tail_loss) if i == 0 else \
                loss + tf.sparse_tensor_to_dense(aligned_tail_loss)

        # Compute head loss

        head_logits = tf.matmul(self.gelu(tf.matmul(inputs, self.head_proj)),
                                self.head_w) + self.head_b  # (sample_num, head_size)
        head_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=head_logits,
                                                                   labels=head_labels)  # (sample_num)
        loss = tf.add(loss, head_loss, name=name)

        return loss

    def softmax(self, inputs, name='softmax'):
        head_logits = tf.matmul(self.gelu(tf.matmul(inputs, self.head_proj)),
                                self.head_w) + self.head_b  # (sample_num, head_size)
        head_softmax = tf.nn.softmax(head_logits)
        softmax_list = [head_softmax[:, :self.cutoff[0]]]
        for i in range(self.cluster_num):
            tail_logits = tf.matmul(self.gelu(tf.matmul(inputs, self.tail_w[i][0])), self.tail_w[i][1]) + \
                          self.tail_w[i][2]
            tail_softmax = tf.nn.softmax(tail_logits)
            index = self.cutoff[0] + i
            softmax_list.append(tail_softmax * head_softmax[:, index:index + 1])
        return tf.concat(softmax_list, axis=1, name=name)

    def gelu(self, input_tensor):
        """Gaussian Error Linear Unit.
        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
        input_tensor: float Tensor to perform activation.
        Returns:
        `input_tensor` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
        return input_tensor * cdf
