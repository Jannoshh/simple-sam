import tensorflow as tf


class SAM():
    def __init__(self, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        self.rho = rho
        self.base_optimizer = base_optimizer

    def first_step(self, gradients, trainable_variables):
        self.e_ws = []
        grad_norm = tf.linalg.global_norm(trainable_variables)
        for i in range(len(trainable_variables)):
            e_w = gradients[i] * self.rho / (grad_norm + 1e-12)
            trainable_variables[i].assign_add(e_w)
            self.e_ws.append(e_w)


    def second_step(self, gradients, trainable_variables):
        for i in range(len(trainable_variables)):
            trainable_variables[i].assign_add(-self.e_ws[i])
        # do the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))

