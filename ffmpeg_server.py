import os
import subprocess
import shutil
import uuid
import json
import sys
import importlib.util
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Check for required dependencies before importing
def check_python_dependencies() -> Tuple[bool, List[str]]:
    """Check if required Python packages are installed."""
    required_packages = {
        'mcp': 'mcp',
        'fastmcp': 'mcp.server.fastmcp'
    }
    
    missing_packages = []
    
    for package_name, import_path in required_packages.items():
        try:
            # Try to find the module spec
            if '.' in import_path:
                parts = import_path.split('.')
                spec = importlib.util.find_spec(parts[0])
                if spec is not None:
                    # Check nested modules
                    try:
                        exec(f"import {import_path}")
                    except ImportError:
                        missing_packages.append(package_name)
                else:
                    missing_packages.append(package_name)
            else:
                spec = importlib.util.find_spec(import_path)
                if spec is None:
                    missing_packages.append(package_name)
        except ImportError:
            missing_packages.append(package_name)
    
    return len(missing_packages) == 0, missing_packages

# Check dependencies before proceeding
dependencies_ok, missing_deps = check_python_dependencies()
if not dependencies_ok:
    print("ERROR: Missing required Python dependencies:")
    for dep in missing_deps:
        print(f"  - {dep}")
    print("\nPlease install the MCP server framework:")
    print("  pip install mcp")
    sys.exit(1)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Enhanced FFmpeg MCP Server for Audio Processing")

# Comprehensive list of audio file extensions
AUDIO_EXTENSIONS = {
    '.wav', '.wave', '.mp3', '.flac', '.ogg', '.oga', '.opus', 
    '.m4a', '.mp4', '.aac', '.wma', '.aiff', '.aif', '.aifc',
    '.caf', '.ape', '.wv', '.tta', '.tak', '.dts', '.ac3',
    '.eac3', '.mka', '.mpc', '.mp2', '.ra', '.rm', '.au',
    '.snd', '.voc', '.w64', '.rf64', '.bwf', '.amb', '.amr',
    '.awb', '.gsm', '.3gp', '.3g2', '.spx', '.oga', '.webm'
}

# Expanded whitelist of FFmpeg flags for audio processing
ALLOWED_FLAGS = {
    # Input/Output
    '-i', '-y', '-n', '-nostdin',
    
    # Audio codec and format
    '-c:a', '-acodec', '-codec:a', '-f', '-format',
    
    # Audio quality and bitrate
    '-b:a', '-ab', '-q:a', '-aq', '-compression_level',
    
    # Sample rate and format
    '-ar', '-sample_rate', '-sample_fmt', '-sample_fmts',
    
    # Channels
    '-ac', '-channels', '-channel_layout', '-guess_layout_max',
    
    # Filters
    '-af', '-filter:a', '-filter_complex', '-filter_script',
    '-filter_complex_script', '-filter_threads',
    
    # Mapping
    '-map', '-map_metadata', '-map_chapters',
    
    # Time and duration
    '-t', '-to', '-ss', '-sseof', '-itsoffset', '-itsscale',
    '-timestamp', '-metadata', '-duration',
    
    # Volume and gain
    '-vol', '-af', 
    
    # Stream selection
    '-vn', '-sn', '-dn', '-an',
    
    # Encoding options
    '-preset', '-profile:a', '-level',
    
    # Logging
    '-loglevel', '-v', '-report', '-hide_banner',
    
    # Advanced audio
    '-async', '-vsync', '-copyts', '-start_at_zero',
    '-copytb', '-shortest', '-dts_delta_threshold',
    
    # Metadata
    '-metadata', '-metadata:s:a', '-id3v2_version',
    
    # Other useful flags
    '-stats', '-progress', '-nostats', '-max_muxing_queue_size',
    '-fflags', '-flags', '-movflags', '-write_id3v1', '-write_id3v2',
    '-write_apetag', '-id3v2_version'
}

def check_docker() -> Tuple[bool, str]:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"], 
            check=True, 
            capture_output=True, 
            text=True
        )
        # Try to run a simple docker command to ensure daemon is running
        subprocess.run(
            ["docker", "ps"], 
            check=True, 
            capture_output=True,
            timeout=5
        )
        return True, f"Docker is available: {result.stdout.strip()}"
    except subprocess.CalledProcessError:
        return False, "Docker is installed but daemon is not running"
    except FileNotFoundError:
        return False, "Docker is not installed"
    except subprocess.TimeoutExpired:
        return False, "Docker daemon is not responding"
    except Exception as e:
        return False, f"Error checking Docker: {str(e)}"

def check_ffmpeg_docker_image() -> Tuple[bool, str]:
    """Check if the FFmpeg Docker image is available."""
    try:
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", "jrottenberg/ffmpeg"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            return True, "FFmpeg Docker image is available"
        else:
            # Try to pull the image
            print("FFmpeg Docker image not found, attempting to pull...")
            pull_result = subprocess.run(
                ["docker", "pull", "jrottenberg/ffmpeg"],
                capture_output=True,
                text=True
            )
            if pull_result.returncode == 0:
                return True, "FFmpeg Docker image pulled successfully"
            else:
                return False, f"Failed to pull FFmpeg Docker image: {pull_result.stderr}"
    except Exception as e:
        return False, f"Error checking FFmpeg Docker image: {str(e)}"

def perform_system_check() -> Dict[str, Any]:
    """Perform comprehensive system dependency check."""
    checks = {
        "python_version": {
            "status": sys.version_info >= (3, 7),
            "message": f"Python {sys.version.split()[0]} (requires >= 3.7)",
            "required": True
        }
    }
    
    # Check Docker
    docker_ok, docker_msg = check_docker()
    checks["docker"] = {
        "status": docker_ok,
        "message": docker_msg,
        "required": True
    }
    
    # Check FFmpeg Docker image if Docker is available
    if docker_ok:
        image_ok, image_msg = check_ffmpeg_docker_image()
        checks["ffmpeg_image"] = {
            "status": image_ok,
            "message": image_msg,
            "required": True
        }
    else:
        checks["ffmpeg_image"] = {
            "status": False,
            "message": "Cannot check FFmpeg image - Docker not available",
            "required": True
        }
    
    # Check Python packages
    deps_ok, missing_deps = check_python_dependencies()
    checks["python_packages"] = {
        "status": deps_ok,
        "message": "All Python packages installed" if deps_ok else f"Missing packages: {', '.join(missing_deps)}",
        "required": True
    }
    
    # Overall status
    all_required_ok = all(check["status"] for check in checks.values() if check["required"])
    
    return {
        "ready": all_required_ok,
        "checks": checks,
        "summary": "System ready for audio processing" if all_required_ok else "System check failed - see details above"
    }

def validate_file_size(file_path: str, max_size_mb: int = 7000) -> Tuple[bool, str]:
    """Validate file size to prevent processing extremely large files."""
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb <= max_size_mb:
            return True, f"File size: {size_mb:.2f} MB"
        else:
            return False, f"File too large: {size_mb:.2f} MB (max: {max_size_mb} MB)"
    except Exception as e:
        return False, f"Error checking file size: {str(e)}"

def get_audio_info(file_path: str) -> Dict[str, Any]:
    """Get audio file information using ffprobe in Docker."""
    try:
        docker_command = [
            "docker", "run", "--rm",
            "-v", f"{os.path.dirname(file_path)}:/work",
            "--network", "none",
            "--user", "nobody",
            "jrottenberg/ffmpeg",
            "ffprobe",  # FIXED: Added missing ffprobe command
            "-hide_banner", "-loglevel", "error",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a:0",
            f"/work/{os.path.basename(file_path)}"
        ]
        
        result = subprocess.run(docker_command, capture_output=True, text=True)
        if result.returncode == 0:
            info = json.loads(result.stdout)
            if info.get('streams'):
                stream = info['streams'][0]
                return {
                    'channels': int(stream.get('channels', 2)),
                    'sample_rate': int(stream.get('sample_rate', 48000)),
                    'codec_name': stream.get('codec_name', 'unknown'),
                    'duration': float(stream.get('duration', 0)),
                    'bit_rate': stream.get('bit_rate', 'unknown')
                }
    except Exception as e:
        print(f"Error getting audio info: {e}")
    
    # Default fallback
    return {
        'channels': 2,
        'sample_rate': 48000,
        'codec_name': 'unknown',
        'duration': 0,
        'bit_rate': 'unknown'
    }

def sanitize_ffmpeg_command(command_template: List[str]) -> Tuple[bool, str]:
    """Sanitize FFmpeg command template for security."""
    allowed_placeholders = {'{input}', '{output}', '{channels}'}
    
    if not isinstance(command_template, list):
        return False, "Command template must be a list of strings."
    
    # Check for {channels} placeholder usage
    channels_usage = [i for i, arg in enumerate(command_template) if '{channels}' in arg]
    if channels_usage:
        # Ensure {channels} is only used as a standalone argument or in numeric context
        for idx in channels_usage:
            if command_template[idx] != '{channels}':
                # If not standalone, check it's used safely (e.g., "channels={channels}")
                if not any(safe_pattern in command_template[idx] for safe_pattern in ['={channels}', ':{channels}']):
                    return False, f"Invalid use of {{channels}} placeholder at position {idx}"
    
    i = 0
    while i < len(command_template):
        arg = command_template[i]
        
        if arg.startswith('-'):
            if arg not in ALLOWED_FLAGS:
                return False, f"Flag {arg} is not allowed."
            
            # Flags that expect a value
            value_flags = {
                '-i', '-af', '-filter:a', '-filter_complex', '-c:a', '-acodec',
                '-codec:a', '-ar', '-sample_rate', '-ac', '-channels', '-f',
                '-format', '-map', '-preset', '-t', '-ss', '-to', '-sseof',
                '-filter', '-filter_script', '-loglevel', '-sample_fmt',
                '-b:a', '-ab', '-q:a', '-aq', '-compression_level', '-vol',
                '-metadata', '-metadata:s:a', '-id3v2_version', '-channel_layout',
                '-profile:a', '-level', '-itsoffset', '-itsscale', '-timestamp',
                '-dts_delta_threshold', '-max_muxing_queue_size', '-fflags',
                '-flags', '-movflags', '-filter_complex_script', '-filter_threads',
                '-guess_layout_max'
            }
            
            if arg in value_flags:
                i += 1
                if i >= len(command_template):
                    return False, f"Flag {arg} expects a value but none found."
                
                val = command_template[i]
                # Check for shell injection characters
                if any(c in val for c in [';', '&&', '||', '`', '$(']):
                    return False, f"Value {val} for flag {arg} contains suspicious characters."
                
                # Allow placeholders and non-empty values
                if val not in allowed_placeholders and not val.strip():
                    return False, f"Value {val} for flag {arg} is invalid."
        else:
            # Non-flag arguments
            if arg not in allowed_placeholders:
                if any(c in arg for c in [';', '&&', '||', '`', '$(']):
                    return False, f"Argument {arg} contains suspicious characters."
        i += 1
    
    # Check for required placeholders
    input_count = sum(1 for x in command_template if '{input}' in x)
    output_count = sum(1 for x in command_template if '{output}' in x)
    
    if input_count != 1 or output_count != 1:
        return False, "Command template must contain exactly one {input} and one {output} placeholder."
    
    return True, "Command template is valid."

def build_fixed_ffmpeg_command(command_template: List[str], channels: int) -> List[str]:
    """Build FFmpeg command with fixed audio output settings."""
    # Fixed options for 48kHz, 32-bit WAV with original channel count
    fixed_options = [
        '-ar', '48000',        # 48kHz sample rate
        '-sample_fmt', 's32',  # 32-bit audio
        '-ac', str(channels),  # preserve original channel count
        '-c:a', 'pcm_s32le'   # ensure PCM codec for WAV
    ]
    
    # Remove any existing audio format options from template
    options_to_remove = {'-ar', '-sample_rate', '-sample_fmt', '-ac', '-channels'}
    filtered_command = []
    skip_next = False
    
    for i, arg in enumerate(command_template):
        if skip_next:
            skip_next = False
            continue
        if arg in options_to_remove:
            skip_next = True
            continue
        filtered_command.append(arg)
    
    # Find output position
    output_index = None
    for i, arg in enumerate(filtered_command):
        if '{output}' in arg:
            output_index = i
            break
    
    # Insert fixed options before output
    if output_index is None:
        filtered_command.extend(fixed_options)
    else:
        filtered_command = (
            filtered_command[:output_index] +
            fixed_options +
            filtered_command[output_index:]
        )
    
    # Ensure output format is WAV
    has_format = any(arg == '-f' for arg in filtered_command)
    if not has_format:
        # Find output position and insert format before it
        for i, arg in enumerate(filtered_command):
            if '{output}' in arg:
                filtered_command.insert(i, '-f')
                filtered_command.insert(i+1, 'wav')
                break
        else:
            # If no {output} found, append at end (shouldn't happen due to validation)
            filtered_command.extend(['-f', 'wav'])
    
    return filtered_command

def run_custom_ffmpeg_in_docker(
    input_dir: str,
    input_filename: str,
    command_template: List[str],
    output_subdir: str = "output",
    output_filename: str = None
) -> Dict[str, Any]:
    """Run FFmpeg command in Docker container."""
    # Sanitize command
    is_valid, message = sanitize_ffmpeg_command(command_template)
    if not is_valid:
        return {
            "stdout": "",
            "stderr": f"Command rejected: {message}",
            "returncode": 1,
            "output_path": None
        }
    
    # Get audio info to preserve channels
    input_path = os.path.join(input_dir, input_filename)
    audio_info = get_audio_info(input_path)
    channels = audio_info['channels']
    
    # Build command with fixed settings
    fixed_command = build_fixed_ffmpeg_command(command_template, channels)
    
    # Create output directory (timestamp already appended by caller)
    output_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    if not output_filename:
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_processed.wav"
    else:
        # Ensure output is WAV
        if not output_filename.lower().endswith('.wav'):
            output_filename = os.path.splitext(output_filename)[0] + '.wav'
    
    # Prepare Docker paths
    container_input = f"/work/{input_filename}"
    container_output = f"/work/{output_subdir}/{output_filename}"
    
    # Replace placeholders
    command_args = []
    for arg in fixed_command:
        arg = arg.replace("{input}", container_input)
        arg = arg.replace("{output}", container_output)
        arg = arg.replace("{channels}", str(channels))
        command_args.append(arg)
    
    # Build Docker command
    docker_command = [
        "docker", "run", "--rm",
        "-v", f"{input_dir}:/work",
        "--network", "none",
        "--user", "nobody",
        "jrottenberg/ffmpeg"
    ] + command_args
    
    # Run command with timeout protection
    try:
        result = subprocess.run(
            docker_command, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout per file
        )
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Processing timeout: FFmpeg took longer than 5 minutes",
            "returncode": -1,
            "output_path": None,
            "audio_info": audio_info
        }
    
    # Don't clean up on failure - keep partial results for debugging
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "output_path": os.path.join(output_dir, output_filename) if result.returncode == 0 else None,
        "audio_info": audio_info
    }

@mcp.tool(description="Run any sanitized ffmpeg command template on all audio files in a directory using Docker with fixed 48kHz 32-bit WAV output.")
def any_ffmpeg_command(
    directory: str,
    command_template: List[str],
    file_extensions: Optional[List[str]] = None,
    output_subdir: str = "output",
    max_file_size_mb: int = 7000
) -> Dict[str, Any]:
    """
    Process audio files with FFmpeg in Docker.
    
    Args:
        directory: Directory containing audio files
        command_template: FFmpeg command with {input} and {output} placeholders
        file_extensions: List of extensions to process (default: all audio formats)
        output_subdir: Subdirectory for output files
        max_file_size_mb: Maximum file size in MB to process (default: 7000)
        
    Returns:
        Dictionary with results including Docker availability and file processing results
    """
    # Check Docker availability first
    docker_available, docker_message = check_docker()
    if not docker_available:
        return {
            "error": f"Docker not available: {docker_message}",
            "docker_status": docker_message,
            "results": []
        }
    
    # Determine which extensions to process
    if file_extensions:
        extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in file_extensions}
    else:
        extensions = AUDIO_EXTENSIONS
    
    # Find all matching audio files
    files = []
    skipped_files = []
    
    for f in os.listdir(directory):
        if any(f.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(directory, f)
            # Validate file size
            size_ok, size_msg = validate_file_size(file_path, max_file_size_mb)
            if size_ok:
                files.append(f)
            else:
                skipped_files.append({"file": f, "reason": size_msg})
    
    if not files and not skipped_files:
        return {
            "docker_status": docker_message,
            "error": f"No audio files found with extensions: {extensions}",
            "results": [],
            "skipped_files": []
        }
    
    # Process each file with progress tracking
    results = []
    total_files = len(files)
    
    # Create a timestamped output directory name
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = f"{output_subdir}_{batch_timestamp}"
    
    print(f"Processing {total_files} audio files...")
    print(f"Output directory: {timestamped_output_dir}/")
    
    for i, file in enumerate(files):
        print(f"Processing {i+1}/{total_files}: {file}")
        
        base_name = os.path.splitext(file)[0]
        output_filename = f"{base_name}_processed.wav"
        
        res = run_custom_ffmpeg_in_docker(
            input_dir=directory,
            input_filename=file,
            command_template=command_template,
            output_subdir=timestamped_output_dir,
            output_filename=output_filename
        )
        
        results.append({
            "input_file": file,
            "output_file": res["output_path"],
            "stdout": res["stdout"],
            "stderr": res["stderr"],
            "success": res["returncode"] == 0,
            "input_info": res["audio_info"]
        })
        
        # Print progress status
        if res["returncode"] == 0:
            print(f"  ✓ Success: {output_filename}")
        else:
            print(f"  ✗ Failed: {file} - {res['stderr'][:100]}")
    
    successful_count = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {successful_count}/{total_files} files processed successfully")
    print(f"Output files saved to: {os.path.join(directory, timestamped_output_dir)}")
    
    return {
        "docker_status": docker_message,
        "output_directory": os.path.join(directory, timestamped_output_dir),
        "timestamped_output_dir": timestamped_output_dir,
        "batch_timestamp": batch_timestamp,
        "processed_count": len(results),
        "successful_count": successful_count,
        "skipped_count": len(skipped_files),
        "results": results,
        "skipped_files": skipped_files
    }

@mcp.tool(description="Get information about available audio processing capabilities.")
def get_audio_processing_info() -> Dict[str, Any]:
    """Get information about the audio processing server capabilities."""
    docker_available, docker_message = check_docker()
    
    return {
        "docker_available": docker_available,
        "docker_status": docker_message,
        "supported_formats": sorted(list(AUDIO_EXTENSIONS)),
        "output_format": "WAV (PCM signed 32-bit little-endian)",
        "output_sample_rate": "48000 Hz",
        "output_bit_depth": "32-bit",
        "channel_handling": "Preserves original channel count",
        "allowed_ffmpeg_flags": sorted(list(ALLOWED_FLAGS)),
        "example_commands": [
            {
                "description": "Simple format conversion",
                "template": ["-i", "{input}", "{output}"]
            },
            {
                "description": "Apply audio filter",
                "template": ["-i", "{input}", "-af", "loudnorm", "{output}"]
            },
            {
                "description": "Trim audio from 10s to 30s",
                "template": ["-i", "{input}", "-ss", "10", "-t", "20", "{output}"]
            },
            {
                "description": "Apply reverb effect",
                "template": ["-i", "{input}", "-af", "aecho=0.8:0.9:1000:0.3", "{output}"]
            },
            {
                "description": "Normalize and compress",
                "template": ["-i", "{input}", "-af", "loudnorm,acompressor=threshold=-20dB:ratio=4:attack=5:release=50", "{output}"]
            }
        ]
    }

@mcp.tool(description="Check system dependencies and readiness for audio processing.")
def check_system_dependencies() -> Dict[str, Any]:
    """Check all system dependencies required for audio processing."""
    return perform_system_check()

if __name__ == "__main__":
    # Perform system check on startup
    print("FFmpeg MCP Server - Checking system dependencies...")
    system_check = perform_system_check()
    
    print("\nSystem Check Results:")
    print("-" * 50)
    for check_name, check_info in system_check["checks"].items():
        status_icon = "✓" if check_info["status"] else "✗"
        print(f"{status_icon} {check_name}: {check_info['message']}")
    print("-" * 50)
    print(f"\n{system_check['summary']}\n")
    
    if not system_check["ready"]:
        print("ERROR: System dependencies not met. Please resolve the issues above.")
        sys.exit(1)
    
    print("Starting MCP server...")
    mcp.run()