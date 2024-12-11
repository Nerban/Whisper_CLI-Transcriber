# Audio Transcription Tool

This is an open-source Python tool for transcribing audio files using [OpenAI Whisper](https://github.com/openai/whisper). It supports multiple output formats including plain text, JSON, SRT, and VTT.

## Features
- Uses OpenAI Whisper's transcription capabilities.
- Supports debugging mode for advanced users.
- Outputs transcription results in multiple formats.
- Handles supported audio formats using FFmpeg.

## Requirements
- Python 3.7+
- FFmpeg installed and available in your system's PATH.
- Whisper Python library installed.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - For Linux:
     ```bash
     sudo apt install ffmpeg
     ```
   - For Mac:
     ```bash
     brew install ffmpeg
     ```
   - For Windows:
     [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.

## Usage
Run the script with the following command:
```bash
python script.py <audio_file> [--debug]
```
- `<audio_file>`: Path to the audio file to be transcribed.
- `--debug` (optional): Enables debug mode to show warnings and detailed output.

### Example:
```bash
python script.py example.mp3 --debug
```

## Output
The tool creates a new folder for each transcription in the `Transkripte` directory. Inside the folder, you will find:
- `transcription.txt`: The plain text transcription.
- `transcription.json`: JSON file with metadata and transcription.
- `transcription.srt`: SubRip subtitle format.
- `transcription.vtt`: WebVTT subtitle format.

## License
This project is licensed under the Apache License 2.0. However, it uses OpenAI Whisper, which is licensed under the MIT License, and FFmpeg, which is licensed under GPL. Ensure compliance with their licenses when redistributing this tool.

## Disclaimer
This tool does not include OpenAI Whisper or FFmpeg binaries. Users are responsible for obtaining and installing these dependencies. Always adhere to the respective licenses when using these tools.

## Contributing
Feel free to open issues or submit pull requests to improve this tool. Contributions are welcome!

## Acknowledgments
- [OpenAI Whisper](https://github.com/openai/whisper) for the transcription model.
- [FFmpeg](https://ffmpeg.org) for audio processing.

For any questions, please contact the repository maintainer.

