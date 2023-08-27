from diffwave.inference import predict as diffwave_predict

model_dir = '/zooper2/esther.whang/mario/diffwave/weights.pt'
spectrogram = '/zooper2/esther.whang/mario/diffwave/audio/reference_0.wav.spec.npy'# get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = diffwave_predict(spectrogram, model_dir, fast_sampling=True)

