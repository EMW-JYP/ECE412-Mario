import re
import os

path = os.path.join(os.getcwd(), "resampled_dataset", "wavs")
wavefiles =  os.listdir(path)
path = os.path.join(os.getcwd(),  "resampled_dataset", "spectrogram_new")
spectrogram_files = os.listdir(path)
#print(wavefiles)

for file in wavefiles:
  #  chunks = re.split('\.',file )
    check_file = file+".spec.npy"
    if check_file not in spectrogram_files:
        print(check_file)
print(len(wavefiles))
print(len(spectrogram_files))





'''
with open('transcriptions.txt', 'r') as file:
    with open("new_transcriptions.txt", 'w') as output:
        for line in file:
         #   print(line)
            chunks = re.split('\|',line)
          #  print(chunks)
            path = re.split("\/",chunks[0])
          #  print(path)
            if path[1] in wavefiles:
                output.write(line)
            else:
                print(line)
            #long_string = "wavs/" + path[1]
            #output.write(long_string+ os.linesep)

'''
'''
with open('transcriptions.txt', 'r') as file:
    with open("audio_paths.txt", 'w') as output:
        for line in file:
            print(line)
            chunks = re.split('\|',line)
            print(chunks)
            path = re.split("\/",chunks[0])
            print(path)
            long_string = "wavs/" + path[1]
            output.write(long_string+ os.linesep)
            '''