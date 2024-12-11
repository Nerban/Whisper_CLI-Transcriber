import os
import sys
import whisper
import subprocess
import json
import warnings
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub.utils import mediainfo

def is_supported_by_ffmpeg(file_path):
    """Check if the file format is supported by ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return "Duration:" in result.stderr  # ffmpeg outputs "Duration" for valid audio files
    except FileNotFoundError:
        print("\033[91m[ERROR]\033[0m ffmpeg is not installed or not available in PATH.")
        sys.exit(1)

def prepare_output_directory(audio_file):
    """Prepare the output directory for transcription results."""
    base_output_dir = os.path.splitext(os.path.basename(audio_file))[0]
    output_dir = os.path.join("Transkripte", base_output_dir)

    # If directory already exists, append a counter
    counter = 1
    original_output_dir = output_dir
    while os.path.exists(output_dir):
        output_dir = f"{original_output_dir}_{counter}"
        counter += 1

    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def perform_diarization(audio_file):
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("\033[91m[ERROR]\033[0m Hugging Face token not found. Please set it in the .env file or environment variables.")
        sys.exit(1)

    """Perform speaker diarization using pyannote.audio."""
    print("\033[94m[INFO]\033[0m Loading diarization model...")
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    duration = get_audio_duration(audio_file)

    print(f"\033[94m[INFO]\033[0m Starting diarization...\n estimated duration: {duration:.2f} seconds")
    diarization = pipeline(audio_file, min_speakers=2, max_speakers=5)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })


    return speaker_segments


def get_audio_duration(audio_file):
    """Retrieve the duration of the audio file in seconds."""
    info = mediainfo(audio_file)
    return float(info['duration'])

def merge_diarization_with_transcription(speaker_segments, transcription_segments):
    """Combine diarization and transcription results."""
    combined_results = []
    for segment in transcription_segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']

        speaker = "Unknown"
        for speaker_segment in speaker_segments:
            if speaker_segment['start'] <= start <= speaker_segment['end']:
                speaker = speaker_segment['speaker']
                break

        combined_results.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text
        })

    return combined_results

def transcribe_audio(audio_file, output_dir, debug=False):
    """Perform transcription using Whisper and save results in multiple formats."""
    print("\033[94m[INFO]\033[0m Loading Whisper model: \033[1mmedium\033[0m")
    if debug:
        model = whisper.load_model("medium")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings
            model = whisper.load_model("medium")

    print("\033[94m[INFO]\033[0m Starting transcription...")
    if debug:
        result = model.transcribe(audio_file, language="de", verbose=False)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings
            result = model.transcribe(audio_file, language="de", verbose=False)
    transcription_segments = result['segments']

    return transcription_segments

def save_combined_results(output_dir, combined_results):
    """Save combined transcription and diarization results to various formats."""
    # Save combined results to JSON
    output_json_file = os.path.join(output_dir, "combined_transcription.json")
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=4)

    # Save to TXT format
    output_text_file = os.path.join(output_dir, "transcription.txt")
    with open(output_text_file, "w", encoding="utf-8") as f:
        for entry in combined_results:
            f.write(f"Speaker {entry['speaker']}: {entry['text']}\n")

    # Save to SRT format
    output_srt_file = os.path.join(output_dir, "transcription.srt")
    with open(output_srt_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(combined_results):
            f.write(f"{i + 1}\n")
            f.write(f"{entry['start']:.3f} --> {entry['end']:.3f}\n")
            f.write(f"Speaker {entry['speaker']}: {entry['text']}\n\n")

    # Save to VTT format
    output_vtt_file = os.path.join(output_dir, "transcription.vtt")
    with open(output_vtt_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for entry in combined_results:
            f.write(f"{entry['start']:.3f} --> {entry['end']:.3f}\n")
            f.write(f"Speaker {entry['speaker']}: {entry['text']}\n\n")

def main():
    load_dotenv()

    # Check if filename is passed as an argument
    if len(sys.argv) < 2:
        print("\033[91m[ERROR]\033[0m Please provide the audio file name as an argument.")
        sys.exit(1)

    # Get audio file name from arguments
    audio_file = sys.argv[1]

    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"\033[91m[ERROR]\033[0m The file \033[1m{audio_file}\033[0m does not exist.")
        sys.exit(1)

    # Check if the file format is supported by ffmpeg
    if not is_supported_by_ffmpeg(audio_file):
        print(f"\033[91m[ERROR]\033[0m The file \033[1m{audio_file}\033[0m is not supported by ffmpeg or is not a valid audio file.")
        sys.exit(1)

    # Debug flag
    debug = "--debug" in sys.argv or "--Debug" in sys.argv

    # Prepare output directory
    output_dir = prepare_output_directory(audio_file)

    print("\033[95m========================================\033[0m")
    print("\033[95m        STARTING AUDIO TRANSCRIPTION        \033[0m")
    print("\033[95m========================================\033[0m")

    # Perform diarization
    speaker_segments = perform_diarization(audio_file)

    # Perform transcription
    transcription_segments = transcribe_audio(audio_file, output_dir, debug=debug)

    # Merge results
    combined_results = merge_diarization_with_transcription(speaker_segments, transcription_segments)

    # Save combined results
    save_combined_results(output_dir, combined_results)

    print("\033[95m========================================\033[0m")
    print("\033[95m          AUDIO TRANSCRIPTION DONE          \033[0m")
    print("\033[95m========================================\033[0m")

if __name__ == "__main__":
    main()
