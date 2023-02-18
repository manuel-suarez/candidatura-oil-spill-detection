# Import
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
#from PIL import Image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from schedulers import SGDRScheduler

# Dataset directory
DATA_DIR = '/home/est_posgrado_manuel.suarez/data/oil-spill-dataset'

# Dataset's configuration
x_train_dir = os.path.join(DATA_DIR, 'train', 'images')
y_train_dir = os.path.join(DATA_DIR, 'train', 'labels_1D')

x_valid_dir = os.path.join(DATA_DIR, 'val', 'images')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'labels_1D')

x_test_dir = os.path.join(DATA_DIR, 'test', 'images')
y_test_dir = os.path.join(DATA_DIR, 'test', 'labels_1D')

# Dataset definition and configuration
# helper function for data visualization
def visualize(figname, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig(figname)


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

        0 - Sea Surface
        1 - Oil Spill
        2 - Look-alike
        3 - Ship
        4 - Land
    """

    CLASSES = ['sea_surface', 'oil_spill', 'look_alike', 'ship', 'land']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.png') for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (480, 360))
        # image = np.array(Image.open(self.images_fps[i]).resize((480, 360)))
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (480, 360))
        # mask = np.array(Image.open(self.masks_fps[i]).convert('L').resize((480, 360)))

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Lets look at data we have
#dataset = Dataset(x_train_dir, y_train_dir, classes=['sea_surface', 'oil_spill', 'look_alike', 'ship', 'land'])
dataset = Dataset(x_train_dir, y_train_dir, classes=['oil_spill', 'sea_surface'])

image, mask = dataset[5] # get some sample
print(image.shape, mask.shape)
visualize(
    "figura1-visualizacion.png",
    image=image,
    oil_spill=mask[..., 0].squeeze(),
    background_mask=mask[..., 1].squeeze(),
)

# Data augmentation
import albumentations as A


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=384, min_width=384, always_apply=True, border_mode=0),
        A.RandomCrop(height=384, width=384, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                # A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 384)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# Aplicamos y visualizamos aumentación de datos
# Lets look at augmented data we have
#dataset = Dataset(x_train_dir, y_train_dir, classes=['sea_surface', 'oil_spill', 'look_alike', 'ship', 'land'], augmentation=get_training_augmentation())
dataset = Dataset(x_train_dir, y_train_dir, classes=['sea_surface', 'oil_spill'], augmentation=get_training_augmentation())

image, mask = dataset[12] # get some sample
print(image.shape, mask.shape)
visualize(
    "figura2-aumentacion.png",
    image=image,
    sea_surface=mask[..., 0].squeeze(),
    oil_spill=mask[..., 1].squeeze(),
    background_mask=mask[..., 2].squeeze(),
)

# Modelo de segmentación
import segmentation_models as sm
sm.set_framework('tf.keras')
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

BACKBONES = [
        # VGG
        'vgg16','vgg19',
        # ResNets
        'resnet18',#'resnet34',#'resnet50',#'resnet101','resnet152',
        # ResNeXt
        'resnext50',#'resnext101',
        # Inception
        'inceptionv3',#'inceptionresnetv2',
        # DenseNet
        'densenet121',#'densenet169','densenet201',
        # SE models
        'seresnet18',#'seresnet34','seresnet50',#'seresnet101','seresnet152','seresnext50','seresnext101','senet154',
        # Mobile Nets
        'mobilenet','mobilenetv2',
        # EfficientNets
        'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb3',#'efficientnetb4','efficientnetb5','efficientnetb6','efficientnetb7',
    ]
BATCH_SIZE = 8
#CLASSES = ['sea_surface', 'oil_spill', 'look_alike', 'ship', 'land']
CLASSES = ['look_alike']
LR = 0.0001
EPOCHS = 50
for BACKBONE in BACKBONES:
    print(80*"=")
    print(BACKBONE)
    print(80*"=")
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    model = sm.FPN(BACKBONE, classes=n_classes, activation=activation, )

    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # define scheduler
    schedule = SGDRScheduler(min_lr=1e-5, max_lr=1e-3, steps_per_epoch=20, lr_decay=0.5)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 2, 0.5]))
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Dataset
    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)
    print(train_dataloader[0][0].shape, train_dataloader[0][1].shape)
    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, 384, 384, 3)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, 384, 384, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    callbacks = [
        keras.callbacks.ModelCheckpoint(f"{BACKBONE}_best_model.h5", save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
        schedule
    ]

    # Training
    # train model
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )

    # Resultados del entrenamiento
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))
    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{BACKBONE}_training_results.png")

    # Verificación
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

    # load best weights
    model.load_weights(f"{BACKBONE}_best_model.h5")

    # Métricas del modelo sobre el conjunto de evaluación
    scores = model.evaluate(test_dataloader)
    print("Backbone: {} Loss: {:.5}".format(BACKBONE, scores[0]))
    for metric, value in zip(metrics, scores[1:]):
        print("Backbone: {} mean {}: {:.5}".format(BACKBONE, metric.__name__, value))

    # Resultados visuales sobre el conjunto de entrenamiento
    n = 5
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    for i in ids:
        image, gt_mask = test_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image)

        visualize(
            f"{BACKBONE}_resultado{i}.png",
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask.squeeze(),
            pr_mask=pr_mask.squeeze(),
        )