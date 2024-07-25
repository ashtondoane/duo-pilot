from VideoSync import VideoSync

if __name__ == "__main__":
    #Video/MEG/Audio should be placed under directory input_folder
    vs = VideoSync(input_folder="/Users/user/VideoSync_NonSubject", output_folder="output/") #Edit to your own path...
    # Grabs all data possible from the input folder, loads it into vs object. Check data with the vs.get_<PARAM>() functions.
    vs.parse_files_from_folder()
    # Extract audio files from videos within input_folder. Ignores videos that have an associated audio file already extracted.
    vs.extract_audio_from_videos()
    #Displays audio that exists within VideoSync.__audio_paths.
    vs.display_pulses(channels='all',tmin=50,tmax=52)