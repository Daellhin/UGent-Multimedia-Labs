import math
import random
import re
from collections import namedtuple
from typing import Any
import cv2

import numpy as np
import scipy.ndimage
import simpleaudio as sa
from matplotlib import pyplot as plt
from scipy.io import wavfile
from functools import reduce
from operator import concat


def karplus_strong(frequency: float, duration: float, fs: int):
	n_samples = int(duration * fs)

	p = int(fs / frequency)
	wavetable = [float(random.randint(-1, 1)) for _ in range(p + 1)]  # Yt
	wavetable_copy = wavetable.copy()

	samples: list[float] = []
	current_sample = 0
	previous_value = 0
	while len(samples) < n_samples:
		wavetable[current_sample] = 0.5 * (wavetable[current_sample] + previous_value)
		samples.append(wavetable[current_sample])
		previous_value = samples[-1]
		current_sample += 1
		current_sample = current_sample % (p + 1)
	return samples, wavetable_copy


def karplus_strong_stretch(frequency: float, duration: float, fs: int, S: float):
	n_samples = int(duration * fs)

	p = int(fs / frequency)
	wavetable = [float(random.randint(-1, 1)) for _ in range(p + 1)]  # Yt
	wavetable_copy = wavetable.copy()

	samples: list[float] = []
	current_sample = 0
	previous_value = 0
	while len(samples) < n_samples:
		r = np.random.binomial(1, 1 - 1 / S)
		if r == 0:
			wavetable[current_sample] = 0.5 * (
				wavetable[current_sample] + previous_value
			)
		samples.append(wavetable[current_sample])
		previous_value = samples[-1]
		current_sample += 1
		current_sample = current_sample % (p + 1)
	return samples, wavetable_copy


def play_audio(samples: list[float], fs: int):
	# Ensure that highest value is in 16-bit range
	audio = samples / np.max(np.abs(samples))
	audio = audio * (2**15 - 1)
	# Convert to 16-bit data
	audio = audio.astype(np.int16)
	# Start playback
	play_obj = sa.play_buffer(audio, 1, 2, fs)
	# Wait for playback to finish before exiting
	play_obj.wait_done()


def plot_audio_with_buffer(
	sound: list[list[float]],
	initial_buffer: list[float],
	duration: float,
	fs: int,
	title="Image",
):
	fig = plt.figure(figsize=(12, 6))
	fig.canvas.manager.set_window_title(title)
	t = np.linspace(0, duration, len(sound))

	# Sound
	plt.subplot(2, 1, 1)
	plt.plot(t, sound)
	plt.title("Karplus-Strong Geluidsgolf")
	plt.xlabel("Tijd (s)")
	plt.ylabel("Amplitude")
	plt.grid(True)

	# Buffer
	plt.subplot(2, 1, 2)
	plt.plot(initial_buffer)
	plt.title("Initiele Vertragingsbuffer")
	plt.xlabel("Tijd (ms)")
	plt.ylabel("Amplitude")
	plt.grid(True)

	plt.tight_layout()
	plt.show()


def bereken_frequentie(noot: str, octaaf: int) -> float:
	noten = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
	C1_freq = 32.703  # Basisfrequentie

	# Bereken aantal halve tonen vanaf C1
	noot_index = noten.index(noot)
	octaaf_verschil = octaaf - 1
	halve_tonen = noot_index + (octaaf_verschil * 12)

	# Bereken frequentie: elke halve toon is een vermenigvuldiging met 2^(1/12)
	sprong = 2 ** (halve_tonen / 12)
	frequentie = C1_freq * sprong

	return round(frequentie, 3)


def generate_scale_waveform(frequencies: list[float], duration: float, fs: int):
	f_min = frequencies[0]
	scale_waveform = []
	for frequency in frequencies:
		S = frequency / f_min
		note_waveform = karplus_strong_stretch(frequency, duration, fs, S)[0]
		scale_waveform.extend(note_waveform)
	return scale_waveform


def synthesize_melody(
	rtttl: str, base_duration: int, base_octave: int, bpm: int, fs: int, fade_in=True
):
	duration_pattern = re.compile("^[0-9]{1,2}")
	pitch_pattern = re.compile("[a-zA-Z]#*")
	octave_pattern = re.compile("[0-9]$")
	special_pattern = re.compile("[.]")

	C2_frequency = bereken_frequentie("C", 2)
	notes = rtttl.split(",")

	waveforms = []
	for note in notes:
		duration_r = duration_pattern.findall(note)
		duration = 1 / (int(duration_r[0]) if len(duration_r) > 0 else base_duration)
		pitch = pitch_pattern.findall(note)[0].upper()
		special_duration = special_pattern.findall(note)
		if special_duration:
			duration *= 1.5
		octave_r = octave_pattern.findall(note)
		octave = int(octave_r[0]) if len(octave_r) > 0 else base_octave

		total_duration = duration * (bpm / 60)  # in seconds s

		waveform = []
		if pitch == "P":
			waveform = [0] * int(total_duration * fs)
		else:
			frequency = bereken_frequentie(pitch, octave)
			S = frequency / C2_frequency
			waveform = karplus_strong_stretch(frequency, total_duration, fs, S)[0]

			if fade_in:
				T_fadein = (60 / bpm) / 8
				fadein_samples = int(T_fadein * fs)
				t = np.linspace(0, T_fadein, fadein_samples)
				fade_in_envelope = 1 - np.exp(-5 * t / T_fadein)
				waveform = np.array(waveform)
				waveform[:fadein_samples] = np.multiply(
					waveform[:fadein_samples], fade_in_envelope
				)
		waveforms.extend(waveform)

	return waveforms


def plot_waveforms(
	sounds: list[list[float]], duration: int, frequencies: list[float], title="Image"
):
	fig = plt.figure(figsize=(10, 7))
	fig.canvas.manager.set_window_title(title)
	t = np.linspace(0, duration, len(sounds[0]))
	for i, (waveform, freq) in enumerate(zip(sounds, frequencies)):
		plt.plot(t, np.array(waveform) + (i + 1), label="{:.2f} Hz".format(freq))
	plt.xlabel("Tijd (ms)")
	plt.ylabel("Amplitude+Octave")
	plt.title("full waveforms")
	plt.show()


def plot_spectrograms(
	sounds: list[list[float]], frequencies: list[float], fs: int, title="Image"
):
	fig = plt.figure(figsize=(10, 10))
	fig.canvas.manager.set_window_title(title)
	for ind, (waveform, freq) in enumerate(zip(sounds, frequencies)):
		plt.subplot(3, 4, ind + 1)
		plt.specgram(waveform, Fs=fs, NFFT=4096, noverlap=3500, cmap="Reds")
		plt.title("{:.2f} Hz".format(freq))
	plt.tight_layout()
	plt.show()


def main():
	frequency = 110  # Hz
	duration = 3  # seconden
	fs = 8000  # Hz

	# Opdracht 1: Synthetiseer 1 toon
	print("-- Opdracht 1 --")
	sound, initial_buffer = karplus_strong(frequency, duration, fs)
	plot_audio_with_buffer(sound, initial_buffer, duration, fs, "Karplus-Strong 110hz")
	play_audio(sound, fs)
	wavfile.write("karplus_strong_110hz.wav", fs, np.array(sound, dtype=np.float32))

	# Opdracht 2: Bereken frequentie van noot
	print("-- Opdracht 2 --")
	print("B4: ", bereken_frequentie("B", 4), "Hz")
	print("F5: ", bereken_frequentie("F", 5), "Hz")
	print("G#6:", bereken_frequentie("G#", 6), "Hz")
	print("D7: ", bereken_frequentie("D", 7), "Hz")

	# Opdracht 3: Synthetiseer 6 octaven van A
	print("-- Opdracht 3 --")
	duration = 2  # seconden
	frequencies = list(map(lambda octaaf: bereken_frequentie("A", octaaf), range(1, 7)))
	print(frequencies)
	sounds = list(map(lambda f: karplus_strong(f, duration, fs)[0], frequencies))
	
	[play_audio(sound, fs) for sound in sounds]
	wavfile.write("note_A_6_octaves.wav", fs, np.array(reduce(concat, sounds, []), dtype=np.float32))
	plot_waveforms(sounds, duration, frequencies, "Note A 6 octaves Waveforms")
	plot_spectrograms(sounds, frequencies, fs, "Note A 6 octaves Spectrograms")

	# Opdracht 4: Synthetiseer 6 octaven van A met stretching
	print("-- Opdracht 4 --")
	duration = 2  # seconden
	frequencies = list(map(lambda octaaf: bereken_frequentie("A", octaaf), range(1, 7)))
	print(frequencies)
	sounds = list(map(lambda zip: karplus_strong_stretch(zip[0], duration, fs, 2**(zip[1]-1))[0], zip(frequencies, range(1,7))))
	
	[play_audio(sound, fs) for sound in sounds]
	wavfile.write("note_A_6_octaves_stretched.wav", fs, np.array(reduce(concat, sounds, []), dtype=np.float32))
	plot_waveforms(sounds, duration, frequencies, "Note A 6 octaves-stretched Waveforms")

	# Opdracht 5: Synthetiseer C major en C minor
	print("-- Opdracht 5 --")
	duration = 0.33
	c_major = ["C", "D","E", "F", "G", "A", "B"]
	c_minor = ["C", "D", "D#", "F", "G", "G#", "A#"]
	frequencies_c_major = list(map(lambda note: bereken_frequentie(note, 3), c_major))
	frequencies_c_minor = list(map(lambda note: bereken_frequentie(note, 3), c_minor))
	c_major_sound = generate_scale_waveform(frequencies_c_major, duration, fs)
	c_minor_sound = generate_scale_waveform(frequencies_c_minor, duration, fs)
	
	play_audio(c_major_sound, fs)
	play_audio(c_minor_sound, fs)
	wavfile.write("C major.wav", fs, np.array(c_major_sound, dtype=np.float32))
	wavfile.write("C minor.wav", fs, np.array(c_minor_sound, dtype=np.float32))

	# Opdracht 6: Genereer soundtrack uit RTTTL string
	print("-- Opdracht 6 --")
	fs = 8000

	base_octave = 2
	base_duration = 4
	bpm = 125
	rtttl_1 = "4e3,8p,8e3,8.g3,8e3,32p,8d3,32p,2c3,4b,4p,4e3,8p,8e3,8g3,32p,8e3,32p,8.d3,8c3,32p,8d3,32p,8.c3,4b,4p,4e3,8p,8e3,8.g3,8e3,32p,8d3,32p,2c3,4b,4p,4e3,8p,8e3,8g3,32p,8e3,32p,8.d3,8c3,32p,8d3,32p,8.c3,4b"
	sound1 = synthesize_melody(rtttl_1, base_duration, base_octave, bpm, fs)
	play_audio(sound1, fs)
	wavfile.write("seven_nation_army.wav", fs, np.array(sound1, dtype=np.float32))

	base_octave = 2
	base_duration = 16
	bpm = 125
	rtttl_2 = "2p,8p,8b,8e.3,g3,8f#3,4e3,8b3,4a.3,4f#.3,8e.3,g3,8f#3,4d3,8f3,2b,8p,8b,8e.3,g3,8f#3,4e3,8b3,4d4,8c#4,4c4,8g#3,8c.4,b3,8a#3,4f#3,8g3,2e3,8p,8g3,4b3,8g3,4b3,8g3,4c4,8b3,4a#3,8f#3,8g.3,b3,8a#3,4a#,8b,2b3,8p"
	sound2 = synthesize_melody(rtttl_2, base_duration, base_octave, bpm, fs)
	play_audio(sound2, fs)
	wavfile.write("harry_potter.wav", fs, np.array(sound2, dtype=np.float32))


if __name__ == "__main__":
	main()
