from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split
import tensorflow as tf

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def _augment_dataset(dataset):
    rng = tf.random.Generator.from_seed(123, alg='philox')
    seed = rng.make_seeds(2)[0]

    def flip(image):
        flipped = tf.image.flip_left_right(image)
        return flipped

    def random_brightness(image):
        image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=seed)
        return image
    
    aug_ds = dataset.map(
        lambda x, y: (random_brightness(x), y))
    
    aug_ds = aug_ds.map(lambda x, y: (flip(x), y))
    
    dataset = dataset.concatenate(aug_ds)
    dataset.shuffle(1000)

    return dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return _augment_dataset(train_dataset), validation_dataset, test_dataset