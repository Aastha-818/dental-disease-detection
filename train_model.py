import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import shutil
from sklearn.model_selection import train_test_split

class DentalModelTrainer:
    def __init__(self, **kwargs):
        """Initialize the Dental Disease Model Trainer"""
        super().__init__(**kwargs)  # Added super().__init__ call
        
        self.base_dir = 'organized_dataset'
        self.train_dir = os.path.join(self.base_dir, 'train')
        self.valid_dir = os.path.join(self.base_dir, 'valid')
        self.test_dir = os.path.join(self.base_dir, 'test')
        self.disease_classes = [
            'Caries',
            'Gingivitis',
            'Hypodontia',
            'Mouth_Ulcer',  
            'Tooth_Discoloration_augmented'
        ]
        self.input_shape = (224, 224, 3)
        self.model = None
        
        # Create necessary directories if they don't exist
        self._create_base_directories()

    def _create_base_directories(self):
        """Create the necessary directory structure"""
        for dir_path in [self.base_dir, self.train_dir, self.valid_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)
            for class_name in self.disease_classes:
                os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

    def organize_dataset(self, source_dir='old/archive'):
        """Organize dataset into train, validation, and test sets"""
        print(f"Organizing dataset from {source_dir}")
        
        # Remove existing organized dataset if it exists
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        
        self._create_base_directories()

        # Split and copy files
        for class_name in self.disease_classes:
            source_class_dir = os.path.join(source_dir, class_name)
            if not os.path.exists(source_class_dir):
                print(f"Warning: Directory not found - {source_class_dir}")
                continue

            images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                print(f"Warning: No images found in {source_class_dir}")
                continue

            print(f"Processing {len(images)} images for {class_name}")

            # Split into train (70%), validation (15%), and test (15%)
            train_images, temp = train_test_split(images, train_size=0.7, random_state=42)
            valid_images, test_images = train_test_split(temp, train_size=0.5, random_state=42)

            # Copy files
            for img_list, dest_dir in [
                (train_images, self.train_dir),
                (valid_images, self.valid_dir),
                (test_images, self.test_dir)
            ]:
                for img in img_list:
                    src_path = os.path.join(source_class_dir, img)
                    dst_path = os.path.join(dest_dir, class_name, img)
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        print(f"Error copying {src_path}: {str(e)}")

    def build_model(self):
        """Build and compile the CNN model"""
        print("Building model...")
        
        model = Sequential([
            # First Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Second Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Third Convolutional Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Fourth Convolutional Block (New)
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Fifth Convolutional Block (New)
            Conv2D(1024, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(1024, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Dense Layers
            Flatten(),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.disease_classes), activation='softmax')
            ])

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model

    def train_model(self, epochs=25, batch_size=32):
        """Train the model with the dataset"""
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        print("Setting up data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation
        valid_datagen = ImageDataGenerator(rescale=1./255)

        print("Creating data generators...")
        
        # Set up the generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_directory(
            self.valid_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]

        print("Starting training...")
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def save_model(self, model_path='dental_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

def main():
    """Main function to run the training pipeline"""
    try:
        print("Initializing DentalModelTrainer...")
        trainer = DentalModelTrainer()
        
        print("Organizing dataset...")
        trainer.organize_dataset()
        
        print("Building model...")
        trainer.build_model()
        
        print("Training model...")
        history = trainer.train_model()
        
        print("Saving model...")
        trainer.save_model()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
