import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, UpSampling2D, \
    Flatten, LeakyReLU, GaussianNoise, Reshape, Conv2DTranspose, \
    Input, Add, ActivityRegularization
from tensorflow.keras import Model, Input, optimizers, layers, models


"""
class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Calculate the size of the output from conv layers
        conv_output_size = self._get_conv_output(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
"""
    

class SimpleCNN(models.Sequential):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        super(SimpleCNN, self).__init__()

        self.add(layers.Conv2D(64, kernel_size=2, padding='same', activation='relu',
                               name='conv1', input_shape=input_shape))
        self.add(layers.MaxPooling2D(pool_size=2, name='maxpool1'))
        self.add(layers.Dropout(0.3, name='drpt1'))

        self.add(layers.Conv2D(32, kernel_size=2, padding='same', activation='relu', name='conv2'))
        self.add(layers.MaxPooling2D(pool_size=2, name='maxpool2'))
        self.add(layers.Dropout(0.3, name='drpt2'))

        self.add(layers.Flatten(name='flatten1'))
        self.add(layers.Dense(256, activation='relu', name='dense1'))
        self.add(layers.Dropout(0.5, name='drpt3'))
        self.add(layers.Dense(num_classes, activation='softmax', name='dense2'))

        self.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
                     


class Autoencoder(models.Sequential):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()

        self.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        self.add(layers.MaxPooling2D((2, 2), padding='same'))
        self.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D((2, 2), padding='same'))
        self.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.add(layers.MaxPooling2D((2, 2), padding='same'))
        
        self.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.add(layers.UpSampling2D((2, 2)))
        self.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
        self.add(layers.UpSampling2D((2, 2)))
        self.add(layers.Conv2D(16, (3, 3), activation='relu'))
        self.add(layers.UpSampling2D((2, 2)))
        self.add(layers.Conv2D(1, (3, 3), activation='linear', padding='same'))



class Generator(Model):
    """Define and compile the residual generator of the CounteRGAN."""
    def __init__(self, input_shape: tuple = (28, 28, 1), residuals: bool = True):
        super().__init__()

        self.input_shape = input_shape
        self.residuals = residuals

        self.conv1 = layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=input_shape)
        self.lrelu1 = layers.LeakyReLU(0.2)
        self.drpt1 = layers.Dropout(0.2)
        self.conv2 = layers.Conv2D(64, (3,3), strides=(2, 2), padding='same')
        self.lrelu2 = layers.LeakyReLU(0.2)
        self.drpt2 = layers.Dropout(0.2)
        self.flatten = layers.Flatten()

        # Deconvolution
        # TODO: n_nodes - ?
        n_nodes = 128 * 7 * 7 # foundation for 7x7 image
        self.dense1 = layers.Dense(n_nodes)
        self.lrelu3 = layers.LeakyReLU(0.2)
        self.reshape1 = layers.Reshape((7, 7, 128))
        # upsample to 14x14
        self.convt1 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')
        self.lrelu4 = layers.LeakyReLU(0.2)
        # upsample to 28x28
        self.convt2 = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')
        self.lrelu5 = layers.LeakyReLU(0.2)
        self.convt3 = layers.Conv2D(1, (7,7), activation='tanh', padding='same')
        self.reshape2 = layers.Reshape((28, 28, 1))
        # these are residuals

        self.reg1 = layers.ActivityRegularization(l1=1e-5, l2=0.0)

    def build(self, input_shape):
        self.conv1.build(input_shape)
        input_shape = self.conv1.compute_output_shape(input_shape)

        self.drpt1.build(input_shape)
        input_shape = self.drpt1.compute_output_shape(input_shape)

        self.conv2.build(input_shape)
        input_shape = self.conv2.compute_output_shape(input_shape)

        self.drpt2.build(input_shape)
        input_shape = self.drpt2.compute_output_shape(input_shape)

        self.flatten.build(input_shape)
        input_shape = self.flatten.compute_output_shape(input_shape)

        self.dense1.build(input_shape)
        input_shape = self.dense1.compute_output_shape(input_shape)

        self.reshape1.build(input_shape)
        input_shape = self.reshape1.compute_output_shape(input_shape)

        self.convt1.build(input_shape)
        input_shape = self.convt1.compute_output_shape(input_shape)

        self.convt2.build(input_shape)
        input_shape = self.convt2.compute_output_shape(input_shape)
        
        self.convt3.build(input_shape)
        input_shape = self.convt3.compute_output_shape(input_shape)

        self.reshape2.build(input_shape)
        input_shape = self.reshape2.compute_output_shape(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.lrelu1(x)
        x = self.drpt1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.drpt2(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.lrelu3(x)
        x = self.reshape1(x)

        x = self.convt1(x)
        x = self.lrelu4(x)
        x = self.convt2(x)
        x = self.lrelu5(x)
        x = self.convt3(x)
        x = self.reshape2(x)

        x = self.reg1(x)
        if self.residuals:
            x = inputs + x
        return x
    

class Discriminator(Model):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.conv1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape)
        self.gaus_noise = layers.GaussianNoise(0.2)
        self.lrelu1 = layers.LeakyReLU(0.2)
        self.drpt1 = layers.Dropout(0.4)
        self.conv2 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.lrelu2 = layers.LeakyReLU(0.2)
        self.drpt2 = layers.Dropout(0.4)
        self.conv3 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.lrelu3 = layers.LeakyReLU(0.2)
        self.drpt3 = layers.Dropout(0.4)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1, activation='sigmoid')

    def build(self, input_shape):
        self.conv1.build(input_shape)
        input_shape = self.conv1.compute_output_shape(input_shape)
        self.gaus_noise.build(input_shape)
        input_shape = self.gaus_noise.compute_output_shape(input_shape)
        self.drpt1.build(input_shape)
        input_shape = self.drpt1.compute_output_shape(input_shape)
        self.conv2.build(input_shape)
        input_shape = self.conv2.compute_output_shape(input_shape)
        self.drpt2.build(input_shape)
        input_shape = self.drpt2.compute_output_shape(input_shape)
        self.conv3.build(input_shape)
        input_shape = self.conv3.compute_output_shape(input_shape)
        self.drpt3.build(input_shape)
        input_shape = self.drpt3.compute_output_shape(input_shape)
        self.flatten.build(input_shape)
        input_shape = self.flatten.compute_output_shape(input_shape)
        self.dense1.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.gaus_noise(x)
        x = self.lrelu1(x)
        #if training:
        x = self.drpt1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)
        #if training:
        x = self.drpt2(x)

        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.drpt3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        return x



class CounteRGAN(Model):
    """
    Combine a generator, discriminator, and fixed classifier into the CounteRGAN.
    """

    def __init__(self, generator, discriminator, classifier):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier

    def initialize(self, input_shape):
        self.build(input_shape)

        self.discriminator.trainable = False
        self.classifier.trainable = False

        countergan_input = Input(shape=input_shape, name='countergan_input')
        x_generated = self.generator(countergan_input)

        model = Model(
            inputs=countergan_input,
            outputs=[self.discriminator(x_generated), self.classifier(x_generated)]
        )

        disc_optimizer = optimizers.Adam(learning_rate=0.0001, beta_1=0.5)  # Discriminatpr optimizer
        self.discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])


        gen_optimizer = optimizers.RMSprop(learning_rate=4e-4, decay=1e-8)  # Generator optimizer
        model.compile(gen_optimizer, loss=["binary_crossentropy", "categorical_crossentropy"])

        return model

    
    def build(self, input_shape):
        #self.generator.build((None, *input_shape))
        #self.discriminator.build((None, *input_shape))
        #print(self.discriminator.summary())
        #input_shape = self.flatten.compute_output_shape(input_shape)
        #self.dense1.build(input_shape)
        self.built = True

    def call(self, inputs):
        x_generated = self.generator(inputs)
        return [self.discriminator(x_generated), self.classifier(x_generated)]



def create_generator(in_shape=(28, 28, 1), residuals=True):
    """Define and compile the residual generator of the CounteRGAN."""
    generator_input = Input(shape=in_shape, name='generator_input')
    generator = Conv2D(64, (3,3), strides=(2, 2), padding='same')(generator_input)
    generator = LeakyReLU(alpha=0.2)(generator)
    generator = Dropout(0.2)(generator)
    generator = Conv2D(64, (3,3), strides=(2, 2), padding='same')(generator)
    generator = LeakyReLU(alpha=0.2)(generator)
    generator = Dropout(0.2)(generator)
    generator = Flatten()(generator)

    # Deconvolution
    n_nodes = 128 * 7 * 7 # foundation for 7x7 image
    generator = Dense(n_nodes)(generator)
    generator = LeakyReLU(alpha=0.2)(generator)
    generator = Reshape((7, 7, 128))(generator)
    # upsample to 14x14
    generator = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(generator)
    # upsample to 28x28
    generator = LeakyReLU(alpha=0.2)(generator)

    generator = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same')(generator)
    generator = LeakyReLU(alpha=0.2)(generator)

    generator = Conv2D(1, (7,7), activation='tanh', padding='same')(generator)
    generator = Reshape((28, 28, 1))(generator)
    # these are residuals

    generator_output = ActivityRegularization(l1=1e-5, l2=0.0)(generator)
    print(generator)
    print(generator_output)

    if residuals:
        generator_output = Add(name="output")([generator_input, generator_output])

    return Model(inputs=generator_input, outputs=generator_output)


def create_discriminator(in_shape=(28, 28, 1)):
    """ Define a neural network binary classifier to classify real and generated
    examples."""
    model = Sequential([
        Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape),
        GaussianNoise(0.2),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),
        Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
        LeakyReLU(alpha=0.2),
        Dropout(0.4),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ], name="discriminator")
    optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_countergan(generator, discriminator, classifier, input_shape=(28, 28, 1)):
    """Combine a generator, discriminator, and fixed classifier into the CounteRGAN."""
    discriminator.trainable = False
    classifier.trainable = False

    countergan_input = Input(shape=input_shape, name='countergan_input')

    x_generated = generator(countergan_input)

    countergan = Model(
        inputs=countergan_input,
        outputs=[discriminator(x_generated), classifier(x_generated)]
    )

    optimizer = optimizers.RMSprop(learning_rate=4e-4, decay=1e-8)  # Generator optimizer
    countergan.compile(optimizer, loss=["binary_crossentropy", "categorical_crossentropy"])
    return countergan



if __name__ == "__main__":
    input_shape = (1, 28,28,1)
    input = tf.random.uniform(shape=input_shape)
    gen = Generator((28, 28, 1))
    prediction = gen.predict(input)
    print(prediction.shape)
    





        




