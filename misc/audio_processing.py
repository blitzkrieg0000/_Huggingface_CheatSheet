import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import numpy as np
from datasets import Audio, load_dataset
from IPython.display import Audio as IPythonAudio
from transformers import AutoFeatureExtractor


def CalculateTimeSeries(audio_array, sampling_rate):
    sec = len(audio_array) / sampling_rate
    timeSeries = np.linspace(0, sec, len(audio_array))
    return timeSeries, sec

def VisualizeAudio(time_series, audio_array, color, override=False):
    if not override:
        plt.figure(figsize=(10, 4))

    plt.plot(time_series, audio_array, color=color)
    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)



rawDataset = load_dataset("PolyAI/minds14", name="en-US", split="train", trust_remote_code=True)
raw_audio_array = rawDataset[0]["audio"]["array"]
sampling_rate = rawDataset[0]["audio"]["sampling_rate"]
rawDatasetTimeSeries, sec = CalculateTimeSeries(raw_audio_array, 8000)
print(raw_audio_array.shape)

VisualizeAudio(rawDatasetTimeSeries, raw_audio_array, "r")


dataset = rawDataset.cast_column("audio", Audio(sampling_rate=16000))
audio_array = dataset[0]["audio"]["array"]
sampling_rate = dataset[0]["audio"]["sampling_rate"]
datasetTimeSeries, sec2 = CalculateTimeSeries(audio_array, sampling_rate)
print(audio_array.shape)

VisualizeAudio(datasetTimeSeries, audio_array, "b")



#? Interactive
IPythonAudio(data=audio_array, rate=sampling_rate) 



#! Feature Extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

features = feature_extractor(
    audio_array, 
    sampling_rate=sampling_rate,
    # padding=True,
    # max_length=100000,
    # truncation=True
)

print(audio_array.min(), audio_array.max())
print(features.input_values[0].min(), features.input_values[0].max())

VisualizeAudio(datasetTimeSeries, features.input_values[0], "g", override=True)
plt.show()


