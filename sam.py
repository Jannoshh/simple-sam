import tensorflow as tf

def get_training_loop(model, loss_object, optimizer, loss_metric=None, accuracy_metric=None):

    @tf.function
    def train_step_SAM(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        rho = 0.05
        grad_norm = tf.linalg.global_norm(model.trainable_variables)
        e_ws = []
        for i in range(len(model.trainable_variables)):
            e_w = gradients[i] * rho / (grad_norm + 1e-12)
            model.trainable_variables[i].assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        for i in range(len(model.trainable_variables)):
            model.trainable_variables[i].assign_add(-e_ws[i])
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if loss_metric:
            loss_metric(loss)
        if accuracy_metric:
            accuracy_metric(labels, predictions)

    return train_step_SAM


