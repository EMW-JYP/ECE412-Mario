
from scipy.io import wavfile
import noisereduce as nr
#not worth it
# load data
rate, data = wavfile.read("wavs/mario2.wav")
# perform noise reduction
print(rate)
print("start")
reduced_noise = nr.reduce_noise(y=data, sr=22050)
print("done")
wavfile.write("wavs/mario2_reduced.wav", rate, reduced_noise)
