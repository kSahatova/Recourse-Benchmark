import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from carla_visual.dataloaders.online_dataset import load_online_dataset



def pytorch_dataset_to_tf_dataset(torch_ds, input_shape, num_classes):
    """
    Converts pytorch dataset into tensorflow dataset
    """
    
    def generator():
        nonlocal torch_ds

        for image, label in torch_ds:
            c, w, h = image.shape
            image = image.numpy().reshape(w, h, c)

            image = tf.convert_to_tensor(image)
            label = tf.convert_to_tensor(label)
            label = to_categorical(label, num_classes=num_classes)

            yield image, label 
        

    tf_dataset = tf.data.Dataset.from_generator(generator, 
                                                output_signature=(
                                                    tf.TensorSpec(shape=(input_shape), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(num_classes,), dtype=tf.int64)
                                                    )
    )
    return tf_dataset



if __name__ == '__main__':

    ds_name = 'MNIST'
    data_root = 'D:\PycharmProjects\XAIRobustness\data\images'

    input_shape = (28, 28, 1)
    num_classes = 10

    torch_ds, _ = load_online_dataset(ds_name, data_root, download=False, split_val=False)
    print(torch_ds)  
    ds = pytorch_dataset_to_tf_dataset(torch_ds, input_shape, num_classes=10)
    print(next(iter(ds))[0].shape)



