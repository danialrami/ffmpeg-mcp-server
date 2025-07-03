# FFmpeg MCP Server for Audio Processing

A secure, containerized MCP (Model Context Protocol) server that provides AI assistants with powerful FFmpeg audio processing capabilities. Process entire directories of audio files with consistent output format while maintaining security through Docker sandboxing.

## Features

- **Universal Audio Format Support**: Processes 40+ audio formats including WAV, MP3, FLAC, OGG, M4A, and more
- **Standardized Output**: All files converted to 48kHz, 32-bit WAV format with preserved channel count
- **Batch Processing**: Process entire directories with progress tracking and detailed reporting
- **Security First**: 
  - All FFmpeg commands run in sandboxed Docker containers
  - Network access disabled during processing
  - Comprehensive command sanitization
  - File size limits (default 7GB)
- **Intelligent Processing**:
  - Automatic audio metadata extraction
  - Channel count preservation
  - Timestamped output directories prevent overwrites
  - Timeout protection (5 minutes per file)
- **MCP Integration**: Seamlessly integrates with AI assistants supporting the Model Context Protocol

## What It's For

This tool is designed for audio professionals, sound designers, and developers who need to:
- Apply consistent processing to large audio libraries
- Convert between audio formats while maintaining quality
- Apply FFmpeg filters and effects safely in batch operations
- Integrate audio processing into AI-powered workflows
- Ensure reproducible, standardized output formats

## Requirements

### System Requirements
- **Python**: 3.7 or higher
- **Docker**: Installed and running
- **MCP Framework**: `pip install mcp`
- **Storage**: Sufficient space for audio files (supports up to 7GB per file)

### Automatic Setup
The server automatically:
- Checks all dependencies on startup
- Pulls the FFmpeg Docker image if not present
- Validates system readiness before processing

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install mcp
   ```

3. **Ensure Docker is installed and running**:
   ```bash
   docker --version
   docker ps
   ```

4. **Run the server**:
   ```bash
   python ffmpeg_server.py
   ```

## Usage

### As an MCP Server

The server exposes three main tools for AI assistants:

#### 1. `any_ffmpeg_command`
Process audio files with custom FFmpeg commands:
```python
{
    "directory": "/path/to/audio/files",
    "command_template": ["-i", "{input}", "-af", "loudnorm", "{output}"],
    "file_extensions": ["wav", "mp3"],  # Optional: defaults to all audio formats
    "output_subdir": "processed",       # Optional: defaults to "output"
    "max_file_size_mb": 7000           # Optional: defaults to 7000 (7GB)
}
```

#### 2. `get_audio_processing_info`
Get information about server capabilities:
- Supported audio formats
- Available FFmpeg flags
- Example command templates
- Current Docker status

#### 3. `check_system_dependencies`
Verify all system dependencies are met:
- Python version
- Docker availability
- FFmpeg image status
- MCP framework

### Example FFmpeg Commands

The server includes several example templates:

```python
# Simple format conversion
["-i", "{input}", "{output}"]

# Audio normalization
["-i", "{input}", "-af", "loudnorm", "{output}"]

# Trim audio (10s to 30s)
["-i", "{input}", "-ss", "10", "-t", "20", "{output}"]

# Apply reverb effect
["-i", "{input}", "-af", "aecho=0.8:0.9:1000:0.3", "{output}"]

# Normalize and compress
["-i", "{input}", "-af", "loudnorm,acompressor=threshold=-20dB:ratio=4:attack=5:release=50", "{output}"]
```

### Output Structure

Processed files are saved in timestamped directories:
```
input_directory/
├── audio1.mp3
├── audio2.flac
├── processed_20250103_143052/
│   ├── audio1_processed.wav
│   └── audio2_processed.wav
└── processed_20250103_151230/
    └── audio1_processed.wav
```

## Security Features

- **Sandboxed Execution**: All FFmpeg commands run in Docker containers with no network access
- **Command Sanitization**: Whitelist-based validation of FFmpeg flags and arguments
- **Shell Injection Protection**: Comprehensive checks for malicious characters
- **Resource Limits**: File size limits and processing timeouts
- **User Isolation**: Containers run as unprivileged users

## Advanced Configuration

### Custom File Size Limits
Adjust the `max_file_size_mb` parameter (default: 7000 MB)

### Processing Timeout
Currently set to 5 minutes per file (hardcoded)

### Supported FFmpeg Flags
The server supports a comprehensive whitelist of audio processing flags including:
- Audio codecs and formats
- Quality and bitrate settings
- Filters and effects
- Metadata handling
- Time-based operations

## Troubleshooting

### Common Issues

1. **"Docker not available"**
   - Ensure Docker Desktop is running
   - Check Docker daemon: `docker ps`

2. **"FFmpeg Docker image not found"**
   - The server will auto-pull the image
   - Manual pull: `docker pull jrottenberg/ffmpeg`

3. **"Missing Python dependencies"**
   - Install MCP: `pip install mcp`

4. **Processing timeouts**
   - Check file sizes and complexity
   - Simplify FFmpeg filters if needed

### Debug Mode
Run with Python's verbose flag for detailed output:
```bash
python -v ffmpeg_mcp_server.py
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style patterns
- Security measures are maintained
- Tests pass (if applicable)
- Documentation is updated

## Acknowledgments

- Built on the MCP (Model Context Protocol) framework
- Uses the excellent `jrottenberg/ffmpeg` Docker image
- Designed for integration with AI assistants and audio processing workflows

## Support

For issues, questions, or contributions:
- Open an issue on the repository
- Check existing documentation
- Ensure system requirements are met

---

**Note**: This tool processes audio files only. Video streams are ignored, and all output is standardized to 48kHz, 32-bit WAV format with original channel count preserved.

Built with 🩷 by LUFS Audio