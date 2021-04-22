import argparse
from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torch.optim as optim
import time

from dataset.NSynthDataSet_RawAudio import NSynthDataSet_RawAudio
from network.PitchGRU import PitchGRU
from training.Train_And_Test import Train_And_Test
from util import plotutil
import util

# Config related
config = util.get_config()
base_data_dir = config['base_data_dir']
total_epoch = config['total_epoch']
sample_rate = int(config['sample_rate'])
batch_size = int(config['batch_size'])
learning_rate = float(config['learning_rate'])
weight_decay = float(config['weight_decay'])
lower_pitch_limit = int(config['lower_pitch_limit'])
upper_pitch_limit = int(config['upper_pitch_limit'])
device = util.get_device()

def load_dataset():
    train_ds = NSynthDataSet_RawAudio('train')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = NSynthDataSet_RawAudio('test')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    validate_ds = NSynthDataSet_RawAudio('validate')
    validate_loader = torch.utils.data.DataLoader(validate_ds, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, validate_loader


def train():
    print('Main params: batch_size, learning_rate -->', batch_size, learning_rate)

    train_loader, test_loader, validate_loader = load_dataset()
    model = PitchGRU()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    model.to(device)
    print(model)

    train_and_test = Train_And_Test(model, optimizer, loss, train_loader, test_loader, validate_loader)

    history_train = {'loss': [], 'acc': []}
    history_test = {'loss': [], 'acc': []}

    max_accuracy = 0.0
    for epoch in range(0, total_epoch):
        start = time.perf_counter()
        train_correct_preds, train_loss = train_and_test.train(epoch)
        timetaken = time.perf_counter() - start
        print('Training epoch {:.0f} time taken - {:.2f} seconds'.format(epoch, timetaken))

        train_accuracy = (float(train_correct_preds) /len(train_loader.dataset)) * 100
        history_train['loss'].append(train_loss)
        history_train['acc'].append(train_accuracy)

        test_correct_preds, test_loss = train_and_test.test()
        test_accuracy = (float(test_correct_preds) /len(test_loader.dataset)) * 100
        history_test['loss'].append(test_loss)
        history_test['acc'].append(test_accuracy)

        if test_accuracy > max_accuracy: #Early stopping. Save the model with max test accuracy
            max_accuracy = test_accuracy
            suffix = '{}-{}-{}-{}-{}'.format(lower_pitch_limit, upper_pitch_limit, learning_rate, batch_size,'max-accuracy')
            model_location = 'models/rnn-pitch-estimation-{}.pt'.format(suffix)
            print('Saving model at test accuracy = {} and train accuracy = '.format(test_accuracy,train_accuracy ))
            train_and_test.save_model(model_location)

        print('\n')
        print('Train accuracy % = {:.2f}% | Test accuracy % = {:.2f}%'.format(train_accuracy, test_accuracy))
        print('Overall training loss = {:.2f} | Overall testing loss = {:.2f}'.format(train_loss, test_loss))
        print('-------------------------------------------------------------------------------------------------------')

    print('-------------------------------------------------------------------------------------------------------')
    print('Training complete')
    print('-------------------------------------------------------------------------------------------------------')
    validate_correct_preds, validate_loss = train_and_test.validate()
    validate_accuracy = (float(validate_correct_preds) /len(validate_loader.dataset)) * 100

    print('\n')
    print('Validate accuracy % = {:.2f}% | Validate loss = {:.2f}'.format(validate_accuracy, validate_loss))
    print('-------------------------------------------------------------------------------------------------------')

    suffix = '{}-{}-{}-{}'.format(lower_pitch_limit, upper_pitch_limit, learning_rate, batch_size)
    model_location = 'models/rnn-pitch-estimation-{}.pt'.format(suffix)
    plot_location = 'models/rnn-pitch-estimation-{}.png'.format(suffix)

    train_and_test.save_model(model_location)
    plotutil.plot_losses(history_train, history_test, plot_location)

    return 'success'


# python main.py --operation=train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', required=True)

    args = parser.parse_args()
    if args.operation == 'train':
        print("Operation: Train")
        train()

    if args.operation == 'predict':
        print("Operation: predict")
