clear; close all; clc;

pathToDatabase = 'Barack_Obama';
DF_type = 'ElevenLabs';
wav_file_dir = fullfile(pathToDatabase, DF_type);

PathToFeatures = 'Matlab_Features';

dinfo = dir( fullfile(wav_file_dir, '*.wav') );
wav_files = {dinfo.name};

for i=1:length(wav_files)
    % disp(wav_files(i))
    filePath = fullfile(wav_file_dir, wav_files(i))
    filename_vec = split(wav_files(i),".");
    filename = filename_vec(end-1);
    % filepath = wav_files(i);

    [x,fs] = audioread(filePath{1});
    disp(fs)
    [stat,delta,double_delta] = extract_lfcc(x,fs,20,1024,20);
    LFCC = [stat delta double_delta]';
    filename_LFCC = fullfile(pathToDatabase, PathToFeatures, horzcat('LFCC_', filename{1}, '.mat'));
    save(filename_LFCC, 'LFCC');
    LFCC = [];
end
disp('Done!');