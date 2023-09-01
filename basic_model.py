from models.model import Model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.experimental import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):

        self.model = models.Sequential()

        self.model.add(layers.Conv2D(8, (3, 3), activation='relu',input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(16, (3, 3), activation='relu',input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu',input_shape=input_shape))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Flatten())

        # classification
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(64, activation='relu'))
        
        self.model.add(layers.Dense(categories_count, activation='softmax'))
    
    def _compile_model(self):
        # Your code goes here

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        
