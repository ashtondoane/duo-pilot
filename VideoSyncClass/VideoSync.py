import numpy as np
import matplotlib.pyplot as plt
from subprocess import run, call
from pathlib import Path
import yaml
import os
from scipy.io import wavfile
from scipy.io.wavfile import read as wavread
import mne

class VideoSync:
    def __init__(self, input_folder=None, output_folder="/output"):
        """
        Summary: VideoSync represents an object performing synchronization between multiple streams of data.

        Attributes:
            input_folder: Directory where data is supplied from.
            output_folder: Directory where analysis files will be published to. Defaults to "input_folder/output"

            self.video_paths: List of RELATIVE paths to video files.
            self.audio_paths: List of RELATIVE paths to audio files.
            self.meg_paths: List of RELATIVE paths to meg files.
        """
        # User defined paths for input/output directories
        self.__input_folder = input_folder
        self.__output_folder = output_folder

        # Paths of data within a folder
        self.__video_paths = []
        self.__audio_paths = []
        self.__meg_paths= []

        # Instance variables representing data


    def get_video_paths(self):
        """
        Returns:
            None: if no path has been defined
            string: List of paths (strings) representing videos
        """
        if len(self.__video_paths) == 0:
            print("video_path has not yet been instantiated. Please use set_video_path or parse_files_from_folder to instantiate this attribute.")
            return None
        return self.__video_paths.copy()
    
    def get_audio_paths(self):
        """
        Returns:
            None: if no path has been defined
            string: List of paths (strings) representing audio files
        """
        if len(self.__audio_paths) == 0:
            print("audio_path has not yet been instantiated. Please use set_audio_path or parse_files_from_folder to instantiate this attribute.")
            return None
        return self.__audio_paths.copy()
    
    def get_meg_path(self):
        """
        Returns:
            None: if path has not yet been defined
            string: Path of the meg file for a given VideoSync object.
        """
        if len(self.__meg_paths) == 0:
            print("meg_path has not yet been instantiated. Please use set_meg_path or parse_files_from_folder to instantiate this attribute.")
            return None
        return self.__meg_paths.copy()

    def get_input_folder(self):
        """
        Returns:
            None: if path has not yet been defined
            string: Path of the input folder for a given VideoSync object.
        """
        if self.__input_folder is None:
            print("input_path has not yet been instantiated. Please use set_input_path to instantiate this attribute.")
            return None
        return self.__input_folder
    
    def get_output_folder(self):
        """
        Returns:
            None: if path has not yet been defined
            string: Path of the output folder for a given VideoSync object.
        """
        if self.__output_folder is None:
            print("output_path has not yet been instantiated. Please use set_output_path to instantiate this attribute.")
            return None
        return self.__output_folder

    def set_video_paths(self, paths):
        """
        Summary: Sets video_paths to a list of paths supplied as arguments.
        """
        temp = []
        if type(paths) != list:
            raise TypeError("Cannot set video paths to given input. Expected list, recieved " + type(paths) + " Paths supplied must be contained within a list in format [s1, s2, ..., sn].")
        for i,item in enumerate(paths):
            if not type(item) == str:
                raise TypeError("Could not set video paths to the provided path. Item " + str(item) + " at index " + str(i) + " is not a string")
            else:
                temp.append(item)
            if not os.path.exists(os.path.join(self.__input_folder, item)):
                print("Warning: \'"+ item + "\' could not be found as an item or directory.")
        self.__video_paths = temp

    def set_audio_paths(self, paths):
        """
        Summary: Sets audio_paths to a list of paths supplied as arguments.
        """
        temp = []
        if type(paths) != list:
            raise TypeError("Cannot set audio paths to given input. Expected list, recieved " + type(paths) + " Paths supplied must be contained within a list in format [s1, s2, ..., sn].")
        for i,item in enumerate(paths):
            if not type(item) == str:
                raise TypeError("Could not set audio paths to the provided path. Item " + str(item) + " at index " + str(i) + " is not a string")
            else:
                temp.append(item)
            if not os.path.exists(os.path.join(self.__input_folder, item)):
                print("Warning: \'"+ item + "\' could not be found as an item or directory.")
        self.__audio_paths = temp

    def set_meg_paths(self, paths):
        """
        Summary: Sets meg_paths to a list of paths supplied as arguments.
        """
        temp = []
        if type(paths) != list:
            raise TypeError("Cannot set meg paths to given input. Expected list, recieved " + type(paths) + " Paths supplied must be contained within a list in format [s1, s2, ..., sn].")
        for i,item in enumerate(paths):
            if not type(item) == str:
                raise TypeError("Could not set meg paths to the provided path. Item " + str(item) + " at index " + str(i) + " is not a string")
            else:
                temp.append(item)
            if not os.path.exists(os.path.join(self.__input_folder, item)):
                print("Warning: \'"+ item + "\' could not be found as an item or directory.")
        self.__meg_paths = temp

    def set_input_folder(self, path):
        """
        Summary: Sets the input folder to the specified path.
        
        Arguments:
            path: The path of the input directory
        """
        if type(path) != str:
            raise TypeError('Path must be provided as a string.')
        if not os.path.exists(path):
            print("Warning: "+ path +" could not be found as a file or directory.")
        self.__input_folder == path

    def set_output_folder(self,path):
        """
        Summary: Sets the output folder to the specified path. This is the location where outputs will be stored. Will create this
                 directory if it does not already exist.
        
        Arguments:
            path: The path of the desired output directory
        """
        if type(path) != str:
            raise TypeError('Path must be provided as a string.')
        if not os.path.exists(path):
            print("Warning: "+ path +" could not be found as a file or directory.")
        self.__output_folder == path

    def parse_files_from_folder(self):
        """
        Summary: If an input folder has been declared, read through this folder and set paths to each of the videos, audio, and 
        """
        files = os.listdir(self.__input_folder)
        audio_paths = []
        video_paths = []
        meg_paths = []
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext == ".wav":
                audio_paths.append(file)
            elif ext == ".fif":
                meg_paths.append(file)
            elif ext == ".mp4":
                video_paths.append(file)
            else:
                pass

            self.set_audio_paths(audio_paths)
            self.set_meg_paths(meg_paths)
            self.set_video_paths(video_paths)

    
    def extract_audio_from_videos(self, channels='all'):
        """
        Summary: Extracts audio tracks from each video requested, and places the audio files in input_folder.

        Arguments:
            channels: Path names of videos that should have their audio extracted. By default, this function will extract from all videos.

        """
        if channels == 'all':
            channels = self.__video_paths
            if len(channels) == 0:
                raise Exception("self.__audio_paths has not been instantiated. Please use set_audio_paths or parse_files_from_folder to populate with paths.") 

        if len(channels) == 0:
            raise Exception("Channels must not be empty.")
        if type(channels) != list:
            raise TypeError("Expected list, recieved " + type(channels) + ". Channels supplied must be contained within a list in format [s1, s2, ..., sn].")

        audio_codecout = 'pcm_s16le'
        audio_suffix = '_16bit'
        for video in channels:
            audio_file = os.path.splitext(video)[0] + audio_suffix + '.wav'
            if os.path.exists(os.path.join(self.__input_folder, audio_file)):
                continue
            video_file = os.path.join(self.__input_folder, video)

            command = ['ffmpeg',
                '-acodec', 'pcm_s24le',       # force little-endian format (req'd for Linux)
                '-i', video_file,
                '-map', '0:a',                # audio only (per DM)
        #         '-af', 'highpass=f=0.1',
                '-acodec', audio_codecout,
                '-ac', '2',                   # no longer mono output, so setting to "2"
                '-y', '-vn',                  # overwrite output file without asking; no video
                '-loglevel', 'error',
                audio_file]
            pipe = run(command, timeout=50)

            if pipe.returncode==0:
                print('Audio extraction was successful for ' + video_file)
                output_path = os.path.join(self.__input_folder, self.__output_folder, audio_file)
                os.renames(audio_file, output_path)
                self.set_audio_paths(self.__audio_paths + [output_path])
                
    
    def display_pulses(self, channels='all', tmin=0, tmax=np.inf):
        """
        Summary: Displays pulses from audio files supplied.

        Parameters:
            channels: Which audio files to show. Defaults to all, which uses self.__audio_paths.
            tmin: Minimum time allowed to show on graph.
            tmax Maximum time allowed to show on graph.
        """
        if channels == 'all':
            channels = self.__audio_paths
            if len(channels) == 0:
                raise Exception("self.__audio_paths has not been instantiated. Please use set_audio_paths or parse_files_from_folder to populate with paths.") 

        if len(channels) == 0:
            raise Exception("Channels must not be empty.")
        if type(channels) != list:
            raise TypeError("Expected list, recieved " + type(channels) + ". Channels supplied must be contained within a list in format [s1, s2, ..., sn].")
        if any(type(c)!=str or not os.path.exists(os.path.join(self.__input_folder, c)) for c in channels):
            raise TypeError("Channels provided must be in valid paths, provided as a string.")
        
        sync_channel = 1; 
        audio_channel = 0
        assert set((sync_channel, audio_channel))=={0,1}
        fig, axset = plt.subplots(len(channels)+1, 1, figsize = [8,6]) #show individual channels seperately, and the 0th plot is the combination of these. 
        for i, audio in enumerate(channels):
            splitName = np.array(audio.split("_")) #name should be split by underscores, check for index that contains word "CAM"
            title = splitName[np.flatnonzero(np.core.defchararray.find(splitName,"CAM")!=-1)]

            srate, wav_signal = wavread(os.path.join(self.__input_folder,audio))
            npts = wav_signal.shape[0]
            # print(f'Numpy array of size {wav_signal.shape} created.')
            # print(f'Sampling rate is {srate} Hz for a total duration of {npts/srate:.2f} seconds.')

            tt = np.arange(npts) / srate
            idx = np.where((tt>=tmin) & (tt<tmax))
            axset[0].plot(tt[idx], wav_signal[idx, sync_channel].T, label=title)
            axset[i+1].plot(tt[idx], wav_signal[idx, sync_channel].T)
            # Make label equal to simply the cam number
            axset[i+1].set_ylabel(title)
        axset[0].set_title("Sync Channels for " + self.__input_folder)
        axset[0].legend()
        plt.show()
    
    def align_pulses(self, channels=[]):
        """
        Summary: 
        """
        pass