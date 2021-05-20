import tensorflow as tf


class SAM():
    def __init__(self, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.rho = rho
        self.base_optimizer = base_optimizer

    def first_step(self, gradients, trainable_variables):
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        for i in range(len(trainable_variables)):
            e_w = gradients[i] * self.rho / (grad_norm + 1e-12)
            trainable_variables[i].assign_add(e_w)
            self.e_ws.append(e_w)


    def second_step(self, gradients, trainable_variables):
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])
        # do the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))


# if you want to use model.fit(), override the train_step method of a model with this function, example is mnist_example_keras_fit.
# for customization see https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/
def sam_train_step(self, data, rho=0.05):
    # Unpack the data. Its structure depends on your model and
    # on what you pass to `fit()`.
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    # Compute gradients
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # first step
    e_ws = []
    grad_norm = tf.linalg.global_norm(gradients)
    for i in range(len(trainable_vars)):
        e_w = gradients[i] * rho / (grad_norm + 1e-12)
        trainable_vars[i].assign_add(e_w)
        e_ws.append(e_w)

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    for i in range(len(trainable_vars)):
        trainable_vars[i].assign_add(-e_ws[i])
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update metrics (includes the metric that tracks the loss)
    self.compiled_metrics.update_state(y, y_pred)
    # Return a dict mapping metric names to current value
    return {m.name: m.result() for m in self.metrics}

