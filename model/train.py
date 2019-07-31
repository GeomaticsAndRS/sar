import argparse
import numpy as np
import pickle
import os
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from model import get_red30_model,PSNR
from keras import losses
from pairgenerator import imgloader

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        """
        @param nb_epochs: total number of epochs for training
        @param initial_lr: the initial learning rate at the beginning of training
        """
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train cnn model for despeckling SAR images",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Image directory for input and target images")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=20,
                        help="total number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="initial learning rate")
    parser.add_argument("--steps", type=int, default=32,
                        help="steps per epoch")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint directory for saving checkpoints")
    parser.add_argument("--model", type=str, default="red30",
                        help="model architecture")                    
    parser.add_argument("--min_date_separation", type=int, default=6,
                       help="Minimum date between image pair acquisition")
    parser.add_argument("--logspace", action="store_true",
                        help="Convert images to logspace before training")
    args = parser.parse_args()

    return args

def get_weights(model):
    """
    Check if the .hd5 file exists.
    """
    args = get_args()
    if args.weight is not None:
        model.load_weights(args.weight)

def checkpoint_directory(checkpoint):
    """
    Check if the checkpoint directory exists.
    """
    if os.path.isdir(checkpoint):
        pass
    else:
        os.mkdir(checkpoint)

def main():
    args = get_args()
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    logspace = args.logspace
    checkpoint = args.output_path

    model = get_red30_model()
    get_weights(model)
    checkpoint_directory(checkpoint)

    opt = Adam(lr=lr)
    callbacks = []

    model.compile(optimizer=opt, loss=losses.mean_squared_error, metrics=[PSNR])

    min_date_separation = args.min_date_separation
    load_dir = args.image_dir
    generator = imgloader(load_dir, batch_size=batch_size, min_date_separation=min_date_separation, logspace=logspace, verbose=False)

    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))

    callbacks.append(ModelCheckpoint(filepath=checkpoint+"/weights.{epoch:03d}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     period=10))

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               callbacks=callbacks,
                               verbose=1)

    return hist

def save_training_history(hist):
    """
    Check if the history directory exists. Save the training history using pickle
    """
    if os.path.exists('history/history') != True:
        os.mknod('history/history')
    with open('history/history', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    print('Training History saved')

if __name__ == '__main__':
    history = main()
    save_training_history(history)
