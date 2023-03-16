
def transfert_learning_mobilenet(X_train, y_train, X_test, y_test,  num_classes, epochs):
    
    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Prepare the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    validation_dataset = validation_dataset.batch(64)
    
    print('\t○\tTransfert Learning avec MobileNet')
    
    # Rescale Pixel Values
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    
    # Create the base model from the pre-trained model MobileNet V2
    IMG_SIZE = (256,256)
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    
    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    # Feature extraction
        #Freeze the convolutionnal base
    base_model.trainable = False
    
        # The base model architecture
    base_model.summary()

        # Add a classification head
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    tf.keras.layers.Dense(1, activation="sigmoid")
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)
    
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)
    
    data_augmentation_mobilenet = tf.keras.Sequential([
      tf.keras.layers.RandomFlip('horizontal'),
      tf.keras.layers.RandomRotation(0.2),
    ])

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = data_augmentation_mobilenet(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    model.summary()

    # Train the model
    initial_epochs = 10
    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)

    return base_model, model, history