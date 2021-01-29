# SAM
Sharpness-Aware Minimization for Efficiently Improving Generalization

## Usage

Using SAM in your training loop is easy:

```python
from sam import SAM
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

model = YourModel()
base_optimizer = tf.keras.optimizers.SGD()  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(base_optimizer)

...

for x, y in dataset:
    train_step_SAM(x, y)
  
...
```

<br>