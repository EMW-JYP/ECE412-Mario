#import numpy as np
#import os
#import re
#import torch

# Importing libraries using import keyword.
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.io import wavfile

import librosa
import librosa.display

#get example, use mamma-mia.wav

sr = 22050
n_fft = 1024
hop_length = 256
n_mels = 80


#test set
filename = '/zooper2/esther.whang/mario/test_dataset/wavs/mario2.wav'
mushroom_kingdom, sr = librosa.load(filename)
print(np.shape(mushroom_kingdom))
# trim silent edges
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\" - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-waveform.png")
plt.savefig(image_path)
plt.close()


#n_fft = 2048
#hop_length = 512
#n_mels = 128
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
print(np.shape(S_DB))
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"\"Mushroom Kingdom, Here We Come\" - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()



mushroom_kingdom_diffwav = np.load('/zooper2/esther.whang/mario/preprocss/mario2.wav.spec.npy')
#S = librosa.feature.melspectrogram(mamma_mia_diffwav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#S_DB = librosa.power_to_db(mamma_mia_diffwav, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(mushroom_kingdom_diffwav, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"\"Mushroom Kingdom, Here We Come\" - Mel Spectrogram with DiffWav")
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-mel-spectrogram-diffwav.png")
plt.savefig(image_path)
plt.close()

mushroom_kingdom_diffwav_reduced = np.load('/zooper2/esther.whang/mario/preprocss/mario2_reduced.wav.spec.npy')
#S = librosa.feature.melspectrogram(mamma_mia_diffwav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#S_DB = librosa.power_to_db(mamma_mia_diffwav, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(mushroom_kingdom_diffwav_reduced, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"\"Mushroom Kingdom, Here We Come\" - Mel Spectrogram with DiffWav, Reduced with NoiseReduce")
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-reduced-mel-spectrogram-diffwav.png")
plt.savefig(image_path)
plt.close()

mushroom_kingdom_diffwav_voicefixed = np.load('/zooper2/esther.whang/mario/preprocss/mario2_voicefixed.wav.spec.npy')
#S = librosa.feature.melspectrogram(mamma_mia_diffwav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#S_DB = librosa.power_to_db(mamma_mia_diffwav, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(mushroom_kingdom_diffwav_voicefixed, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\" - Mel Spectrogram with DiffWav, Reduced with SoundFixer")
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-voicefixed-mel-spectrogram-diffwav.png")
plt.savefig(image_path)
plt.close()


mamma_mia_talknet = np.load('/zooper2/esther.whang/mario/test_dataset/mario2.wav.spec.npy')
#S = librosa.feature.melspectrogram(mamma_mia_talknet, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#S_DB = librosa.power_to_db(mamma_mia_talknet, ref=np.max)
print("check talknet: ", np.shape(mamma_mia_talknet))
plt.figure(figsize=(10,6))
librosa.display.specshow(np.squeeze(mamma_mia_talknet), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"\"Mushroom Kingdom, Here We Come\" - Mel Spectrogram with TalkNet")
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-mel-spectrogram-talknet.png")
plt.savefig(image_path)
plt.close()

#compare iters
#1000
filename = '/zooper2/esther.whang/mario/test_dataset/model_iter_compare/mario2_output_1000.wav'
mushroom_kingdom, sr = librosa.load(filename)
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\", 1000 steps - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-1000-waveform.png")
plt.savefig(image_path)
plt.close()
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\", 1000 steps - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-1000-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()

#5000
filename = '/zooper2/esther.whang/mario/test_dataset/model_iter_compare/mario2_output_5000.wav'
mushroom_kingdom, sr = librosa.load(filename)
print(np.shape(mushroom_kingdom))
# trim silent edges
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\", 5000 steps - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-5000-waveform.png")
plt.savefig(image_path)
plt.close()
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\", 5000 steps - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-5000-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()

#8000
filename = '/zooper2/esther.whang/mario/test_dataset/model_iter_compare/mario2_output_8000.wav'
mushroom_kingdom, sr = librosa.load(filename)
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\", 8000 steps - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-8000-waveform.png")
plt.savefig(image_path)
plt.close()
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\", 8000 steps - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-8000-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()

#10000
filename = '/zooper2/esther.whang/mario/test_dataset/model_iter_compare/mario2_output_10000.wav'
mushroom_kingdom, sr = librosa.load(filename)
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\", 10000 steps - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-10000-waveform.png")
plt.savefig(image_path)
plt.close()
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
print(np.shape(S_DB))
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\", 10000 steps - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-10000-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()

#15000
filename = '/zooper2/esther.whang/mario/test_dataset/model_iter_compare/mario2_output_15000.wav'
mushroom_kingdom, sr = librosa.load(filename)
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\", 15000 steps - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-15000-waveform.png")
plt.savefig(image_path)
plt.close()
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\", 15000 steps - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-15000-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()

#25000
filename = '/zooper2/esther.whang/mario/test_dataset/model_iter_compare/mario2_output_25000.wav'
mushroom_kingdom, sr = librosa.load(filename)
print(np.shape(mushroom_kingdom))
# trim silent edges
plt.figure(figsize=(10,6))
librosa.display.waveplot(mushroom_kingdom, sr=sr)
plt.title(f" \"Mushroom Kingdom, Here We Come\", 25000 steps - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-25000-waveform.png")
plt.savefig(image_path)
plt.close()
S = librosa.feature.melspectrogram(mushroom_kingdom, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f" \"Mushroom Kingdom, Here We Come\", 25000 steps - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mushroom_kingdom-25000-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()

###training set

filename = '/zooper2/esther.whang/mario/wavs/mamma-mia.wav'
mamma_mia, sr = librosa.load(filename)
print(np.shape(mamma_mia))
# trim silent edges
plt.figure(figsize=(10,6))
librosa.display.waveplot(mamma_mia, sr=sr)
plt.title(f"mamma mia - wavefile")
plt.ylabel('Amplitude')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mamma-mia-waveform.png")
plt.savefig(image_path)
plt.close()

#n_fft = 2048
#hop_length = 512
#n_mels = 128
S = librosa.feature.melspectrogram(mamma_mia, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
print(np.shape(S_DB))
plt.figure(figsize=(10,6))
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"mamma mia - Mel Spectrogram with Librosa")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mamma-mia-mel-spectrogram-librosa.png")
plt.savefig(image_path)
plt.close()



mamma_mia_diffwav = np.load('/zooper2/esther.whang/mario/preprocss/mamma-mia.wav.spec.npy')
#S = librosa.feature.melspectrogram(mamma_mia_diffwav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#S_DB = librosa.power_to_db(mamma_mia_diffwav, ref=np.max)
print("check diffwav: ", np.shape(mamma_mia_diffwav))
plt.figure(figsize=(10,6))
librosa.display.specshow(mamma_mia_diffwav, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"mamma mia - Mel Spectrogram with DiffWav")
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mamma-mia-mel-spectrogram-diffwav.png")
plt.savefig(image_path)
plt.close()


mamma_mia_talknet = np.load('/zooper2/esther.whang/mario/resampled_dataset/combined/mamma-mia.wav.spec.npy')
#S = librosa.feature.melspectrogram(mamma_mia_talknet, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
#S_DB = librosa.power_to_db(mamma_mia_talknet, ref=np.max)
print("check talknet: ", np.shape(mamma_mia_talknet))
plt.figure(figsize=(10,6))
librosa.display.specshow(np.squeeze(mamma_mia_talknet), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel');
plt.colorbar(format='%+2.0f dB')
plt.title(f"mamma mia - Mel Spectrogram with TalkNet")
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/presentation_images/mamma-mia-mel-spectrogram-talknet.png")
plt.savefig(image_path)
plt.close()



"""#sample_rate, samples = wavfile.read('/zooper2/esther.whang/mario/wavs/mamma-mia.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
print(np.shape(spectrogram))
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.title(f"mamma mia - Spectrogram with Scipy")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
image_path = os.path.join(os.getcwd(), "/zooper2/esther.whang/mario/mamma-mia-spectrogram-scipy.png")
plt.savefig(image_path)
plt.close()
"""


"""# Generating an array of values
Time_Array = np.linspace(0, 5, math.ceil(5 / Time_difference))
# Actual data array which needs to be plot
Data = 20*(np.sin(3 * np.pi * Time_Array))
# Matplotlib.pyplot.specgram() function to
# generate spectrogram
plt.specgram(Data, Fs=6, cmap="rainbow")

# Set the title of the plot, xlabel and ylabel
# and display using show() function
plt.title('Spectrogram Using matplotlib.pyplot.specgram() Method')
plt.xlabel("DATA")
plt.ylabel("TIME")
plt.show()"""


"""
spectrogram_path = os.path.join(os.getcwd(),  "spectrogram_new")
spectrogram_files = os.listdir(spectrogram_path)
#print(wavefiles)
#rescale https://github.com/lmnt-com/diffwave/issues/14

for file in spectrogram_files:
    print(file)
    chunks = re.split('\.',file )
    filename = chunks[0]+".wav"
    file_path = os.path.join(os.getcwd(),  "spectrogram_new", file)
    spectrogram = np.load(file_path)
    spectrogram = torch.from_numpy(spectrogram)
    spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    print(spectrogram.shape)
    save_path = os.path.join(os.getcwd(),  "resampled_dataset", "spectrogram_new", f'{filename}.spec.npy')
    np.save(save_path, spectrogram.cpu().numpy())"""