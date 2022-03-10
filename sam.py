import tensorflow as tf


class SAM():
    def __init__(self, base_optimizer, rho=0.05, eps=1e-12):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.rho = rho
        self.eps = eps
        self.base_optimizer = base_optimizer

    def first_step(self, gradients, trainable_vars):
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        ew_multiplier = self.rho / (grad_norm + self.eps)
        for i in range(len(trainable_vars)):
            e_w = tf.math.multiply(gradients[i], ew_multiplier)
            trainable_vars[i].assign_add(e_w)
            self.e_ws.append(e_w)

    def second_step(self, gradients, trainable_variables):
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])
        # do the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))


# if you want to use model.fit(), override the train_step method of a model with this function, example is mnist_example_keras_fit.
# for customization see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
def sam_train_step(self, data, rho=0.05, eps=1e-12):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    if len(data) == 3:
        x, y, sample_weight = data
    else:
        sample_weight = None
        x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # first step
    e_ws = []
    grad_norm = tf.linalg.global_norm(gradients)
    ew_multiplier = rho / (grad_norm + eps)
    for i in range(len(trainable_vars)):
        e_w = tf.math.multiply(gradients[i], ew_multiplier)
        trainable_vars[i].assign_add(e_w)
        e_ws.append(e_w)

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    for i in range(len(trainable_vars)):
        trainable_vars[i].assign_sub(e_ws[i])
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics.
    # Metrics are configured in `compile()`.
    self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

    # Return a dict mapping metric names to current value.
    # Note that it will include the loss (tracked in self.metrics).
    return {m.name: m.result() for m in self.metrics}

