import os
import sys
import whisper
import subprocess
import json
import warnings

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

    # Save text transcription
    output_text_file = os.path.join(output_dir, "transcription.txt")
    with open(output_text_file, "w", encoding="utf-8") as f:
        f.write(result["text"])

    # Save JSON format
    output_json_file = os.path.join(output_dir, "transcription.json")
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # Save SRT file
    output_srt_file = os.path.join(output_dir, "transcription.srt")
    with open(output_srt_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result['segments']):
            f.write(f"{i + 1}\n")
            f.write(f"{segment['start']:.3f} --> {segment['end']:.3f}\n")
            f.write(f"{segment['text']}\n\n")

    # Save VTT file
    output_vtt_file = os.path.join(output_dir, "transcription.vtt")
    with open(output_vtt_file, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for segment in result['segments']:
            f.write(f"{segment['start']:.3f} --> {segment['end']:.3f}\n")
            f.write(f"{segment['text']}\n\n")

    print(f"\033[92m[SUCCESS]\033[0m Transcription saved successfully in \033[1m{output_dir}\033[0m")

def main():
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

    # Perform transcription
    transcribe_audio(audio_file, output_dir, debug=debug)

    print("\033[95m========================================\033[0m")
    print("\033[95m          AUDIO TRANSCRIPTION DONE          \033[0m")
    print("\033[95m========================================\033[0m")

if __name__ == "__main__":
    main()
