from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

# Single model.fit call
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=callbacks,
    epochs=30,
    verbose=1
)

val_results = evaluate_model_with_auc(model, val_ds, "Validation")  
test_results = evaluate_model_with_auc(model, test_ds, "Test") 