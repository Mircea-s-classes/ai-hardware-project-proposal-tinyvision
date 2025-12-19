import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, SeparableConv2D, MaxPooling2D, 
                                      BatchNormalization, GlobalAveragePooling2D,
                                      SpatialDropout2D)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2

EPOCHS = args.epochs or 180
LEARNING_RATE = args.learning_rate or 0.0005
BATCH_SIZE = args.batch_size or 32
ENSURE_DETERMINISM = args.ensure_determinism
SPATIAL_DROPOUT_RATE = 0.10  # Adam Wiącek's best setting

# ============================================================================
# DATA AUGMENTATION - Using ranges from Adam Wiącek’s FER Approach
# ============================================================================
def augment_image(image, label):
    # Rotation: ±30 degrees
    angle = tf.random.uniform([], -0.5236, 0.5236)  # 30° in radians
    image = tfa.image.rotate(image, angle, fill_mode='nearest')
    
    # Zoom: range 0.2 (0.8 to 1.2)
    zoom = tf.random.uniform([], 0.8, 1.2)
    h, w, _ = image.shape
    new_h = tf.cast(h * zoom, tf.int32)
    new_w = tf.cast(w * zoom, tf.int32)
    image = tf.image.resize(image, (new_h, new_w))
    image = tf.image.resize_with_crop_or_pad(image, h, w)
    
    # Brightness: 0.2 to 1.2
    brightness_factor = tf.random.uniform([], 0.2, 1.2)
    image = image * brightness_factor
    
    # Horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Clip to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE*8)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

model = Sequential(name='FER_Hybrid_AuthorArchitecture_Fast')

# ============================================================================
# BLOCK 1: Initial Feature Extraction (32 filters)
# First conv stays regular Conv2D (better for raw grayscale input)
# Second conv uses SeparableConv2D for speed
# ============================================================================
model.add(Conv2D(
    32,
    kernel_size=(3, 3),
    strides=1,
    padding='same',
    activation='relu',
    kernel_regularizer=l2(0.0001),
    input_shape=(48, 48, 1),
    name='conv1_1'
))
model.add(BatchNormalization(name='bn1_1'))

model.add(SeparableConv2D(
    32,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv1_2'
))
model.add(BatchNormalization(name='bn1_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1'))  # 24x24
model.add(SpatialDropout2D(SPATIAL_DROPOUT_RATE, name='spatial_dropout1'))

# ============================================================================
# BLOCK 2: Pattern Learning (64 filters) - SeparableConv2D
# ============================================================================
model.add(SeparableConv2D(
    64,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv2_1'
))
model.add(BatchNormalization(name='bn2_1'))

model.add(SeparableConv2D(
    64,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv2_2'
))
model.add(BatchNormalization(name='bn2_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='pool2'))  # 12x12
model.add(SpatialDropout2D(SPATIAL_DROPOUT_RATE, name='spatial_dropout2'))

# ============================================================================
# BLOCK 3: Deep Feature Extraction (96 filters) - SeparableConv2D
# ============================================================================
model.add(SeparableConv2D(
    96,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv3_1'
))
model.add(BatchNormalization(name='bn3_1'))

model.add(SeparableConv2D(
    96,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv3_2'
))
model.add(BatchNormalization(name='bn3_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='pool3'))  # 6x6
model.add(SpatialDropout2D(SPATIAL_DROPOUT_RATE, name='spatial_dropout3'))

# ============================================================================
# BLOCK 4: Final Feature Refinement (128 filters) - SeparableConv2D
# ============================================================================
model.add(SeparableConv2D(
    128,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv4_1'
))
model.add(BatchNormalization(name='bn4_1'))

model.add(SeparableConv2D(
    128,
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    depthwise_regularizer=l2(0.0001),
    pointwise_regularizer=l2(0.0001),
    name='sepconv4_2'
))
model.add(BatchNormalization(name='bn4_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='pool4'))  # 3x3
model.add(SpatialDropout2D(SPATIAL_DROPOUT_RATE, name='spatial_dropout4'))

# ============================================================================
# CLASSIFIER HEAD - Adam Wiącek's Design + Unconventional Dense Layer for Learning
# ============================================================================
model.add(GlobalAveragePooling2D(name='global_avg_pool'))
model.add(Dropout(0.5, name='dropout_gap'))

model.add(Dense(
    128,
    activation='relu',
    kernel_regularizer=l2(0.0001),
    name='fc1'
))
model.add(Dropout(0.5, name='dropout_fc'))

model.add(Dense(
    classes,
    activation='softmax',
    kernel_regularizer=l2(0.0001),
    name='predictions'
))

model.summary()

# ============================================================================
# OPTIMIZER AND CALLBACKS
# ============================================================================

opt = Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    decay=1e-6
)

callbacks.append(BatchLoggerCallback(
    BATCH_SIZE, 
    train_sample_count, 
    epochs=EPOCHS, 
    ensure_determinism=ENSURE_DETERMINISM
))

callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-7,
    verbose=1,
    mode='min'
))

callbacks.append(tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=40,
    restore_best_weights=True,
    verbose=1,
    mode='min'
))

callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
))

model.compile(
    loss='categorical_crossentropy', 
    optimizer=opt, 
    metrics=['accuracy']
)

# Class weights for imbalanced dataset
class_weights = ei_tensorflow.training.get_class_weights(Y_train)
dampened_weights = {k: 1 + 0.3*(v-1) for k, v in class_weights.items()}

# Train
model.fit(
    train_dataset, 
    epochs=EPOCHS, 
    validation_data=validation_dataset, 
    verbose=2, 
    callbacks=callbacks, 
    class_weight=dampened_weights
)

disable_per_channel_quantization = False