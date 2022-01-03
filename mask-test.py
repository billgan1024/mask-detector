import tensorflow as tf
import tensorflow.keras as keras
model = keras.models.load_model("model.h5")
print(model.summary())

# mask was trained with this encoding:
['mask_weared_incorrect', 'with_mask', 'without_mask']
