# simple-SAM
Sharpness-Aware Minimization for Efficiently Improving Generalization

-----------

This is an **unofficial** repository for [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412).

The implementation is in Tensorflow 2 and is heavily inspired by [davda54's PyTorch implementation](https://github.com/davda54/sam).

## Usage

Using SAM is easy in custom training loops:

```python
...

from sam import SAM

model = YourModel()
base_optimizer = tf.keras.optimizers.SGD()  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(base_optimizer)

...

@tf.function
def train_step_SAM(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.first_step(gradients, model.trainable_variables)

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.second_step(gradients, model.trainable_variables)

...

for x, y in dataset:
    train_step_SAM(x, y)
  
...
```

<br>
