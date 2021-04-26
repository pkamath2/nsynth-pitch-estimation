import argparse
import matplotlib
from matplotlib.pyplot import plot
import torch
import torch.nn as nn
import torch.optim as optim
import time
import librosa
import numpy as np
import torchcrepe
import torchaudio
from celluloid import Camera
import matplotlib.pyplot as plt

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
sample_length = int(config['sample_length'])
batch_size = int(config['batch_size'])
learning_rate = float(config['learning_rate'])
weight_decay = float(config['weight_decay'])
lower_pitch_limit = int(config['lower_pitch_limit'])
upper_pitch_limit = int(config['upper_pitch_limit'])
classes = [x for x in range(lower_pitch_limit, upper_pitch_limit)]
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

def predict(sample_file, target_midi): # data is in numpy

    train_loader, test_loader, validate_loader = load_dataset()
    model = PitchGRU()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    model.to(device)

    train_and_test = Train_And_Test(model, optimizer, loss, train_loader, test_loader, validate_loader)
    model = train_and_test.load_model('models/rnn-pitch-estimation-21-81-0.001-32-max-accuracy.pt')

    target_pitch = librosa.midi_to_hz(target_midi)
    print('Target = ', target_pitch)
    print('-----------------------------------------------------------')
    print('Begin predictions - ')
    padded_sample = np.zeros(64000)
    fig, ax = plt.subplots(figsize=(15, 15))
    camera = Camera(fig)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.xlim(0, 64000)
    def numfmt(x, pos): # your custom formatter function: divide by 16000.0
        s = '{}'.format(x / 16000.0)
        return s

    yfmt = matplotlib.ticker.FuncFormatter(numfmt)
    plt.gca().xaxis.set_major_formatter(yfmt)

    for i in range(31):
        start_ts = i * 0.128

        data,_ = librosa.load(sample_file, sr=16000, offset=start_ts, duration=0.128)
        audio_data_stft = librosa.stft(data, n_fft=(2048-1)*2)
        sample = np.concatenate([np.reshape(data, (2048, 1)) * 2, np.abs(audio_data_stft),np.angle(audio_data_stft)], axis=1)
        sample = torch.from_numpy(sample)
        sample = sample.float()
        sample = sample.to(device)
        sample = sample.view(-1, sample.shape[0], 7) # batch_size X 2048 X 1

        prediction = -1
        model.eval()
        with torch.no_grad():
            output, hidden = model(sample) # Output Shape = batch_size, 1, 88 (num pitches)
            output = output[:, int(sample_length/2), :]
            output = output.view(-1, len(classes))
            prediction = librosa.midi_to_hz(classes[output.argmax(dim=1)])
        
        pyin_f0, _, _ = librosa.pyin(data, fmin=librosa.note_to_hz('C0'), fmax=librosa.note_to_hz('C8'), sr=16000, frame_length=1000, hop_length=1000)
        pyin_f0 = np.nan_to_num(pyin_f0)
        pyin_f0 = np.max(pyin_f0)

        data_tensor = torch.from_numpy(data).view(1, -1)
        torchaudio_f0 = torchaudio.functional.detect_pitch_frequency(data_tensor, sample_rate=16000, frame_time=0.008, freq_low=40).item()
        
        crepe_f0 = torchcrepe.predict(data_tensor,
                           16000,
                           2048,
                           40,
                           3400,
                           'full',
                           batch_size=1,
                           device=device)
        crepe_f0 = np.mean(crepe_f0.numpy())
        print('At start time {:.2f} seconds - GRU Prediction={:.2f}Hz; pYIN={:.2f}Hz; torchaudio={:.2f}Hz; Torch Crepe={:.2f}Hz'.format(start_ts, prediction, pyin_f0, torchaudio_f0, crepe_f0))

        padded_sample[i*sample_length:(i*sample_length)+sample_length] = data
        
        plt.plot(padded_sample,color='blue', alpha=0.5)
        ax.text(0.1, 1.01, " ", transform=ax.transAxes,fontsize='x-large')
        
        ax.text(0.1, 1.01, "Target pitch = {:.2f}Hz, \
                            \nGRU predicted pitch = {:.2f}Hz, \
                            \npYIN = {:.2f}Hz, \
                            \nTorch Audio = {:.2f}Hz, \
                            \nTorch Crepe = {:.2f}Hz, ".format(target_pitch, prediction, pyin_f0, torchaudio_f0, crepe_f0), transform=ax.transAxes,fontsize='xx-large')
        camera.snap()
    animation = camera.animate()
    animation.save('animation-{}.gif'.format(target_midi))

    print('-------------------------------------------------------------')



def validate():
    train_loader, test_loader, validate_loader = load_dataset()
    model = PitchGRU()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    model.to(device)

    train_and_test = Train_And_Test(model, optimizer, loss, train_loader, test_loader, validate_loader)
    model = train_and_test.load_model('models/rnn-pitch-estimation-21-81-0.001-32-max-accuracy.pt')
    validate_correct_preds, validate_loss = train_and_test.validate()
    validate_accuracy = (float(validate_correct_preds) /len(validate_loader.dataset)) * 100
    print('\n')
    print('Validate accuracy % = {:.2f}% | Validate loss = {:.2f}'.format(validate_accuracy, validate_loss))
    print('-------------------------------------------------------------------------------------------------------')

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
        predict('/home/purnima/appdir/Github/DATA/NSynth/nsynth-test/audio/guitar_acoustic_021-053-075.wav', 53)
    
    if args.operation == 'validate':
        print("Operation: validate")
        validate()
    