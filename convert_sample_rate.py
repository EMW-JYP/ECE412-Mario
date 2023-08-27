import librosa
import os
import soundfile


final_sr = 22050
wav_path = os.path.join(os.getcwd(), "test_dataset", "wavs_before_sampling")
paths = os.listdir(wav_path)
for line in paths:
       # print(line)
       audio_path = os.path.join(wav_path, line)
        #print(audio_path)
       y, sr = librosa.load(audio_path)
       print(sr)
       clean_audio_path = os.path.join(os.getcwd(), "test_dataset", "wavs", line)
       soundfile.write(clean_audio_path, y, final_sr)