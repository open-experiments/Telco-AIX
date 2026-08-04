# Author: Extract fine-tuned model from compressed archive
# This extracts the fine-tuned model files for containerization
#
import os
import sys
import tarfile
from pathlib import Path

# Configuration
compressed_file = os.environ.get("COMPRESSED_MODEL", "qwen-finetuned-model.tar.xz")
target_model_dir = os.environ.get("MODEL_DIR", "/models")

print(f"Extracting fine-tuned model from: {compressed_file}")
print(f"Target directory: {target_model_dir}")

def extract_model_files():
    """Extract fine-tuned model files from compressed archive"""
    compressed_path = Path(compressed_file)
    target_path = Path(target_model_dir)
    
    # Check if compressed file exists
    if not compressed_path.exists():
        print(f"Error: Compressed file does not exist: {compressed_path}")
        sys.exit(1)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract the tar file
        print("Extracting compressed model archive...")
        with tarfile.open(compressed_path, 'r:xz') as tar:
            # Extract safely: the 'data' filter (Python 3.12+/backported to
            # 3.8.17+) rejects absolute paths, path traversal ('..'), links
            # escaping the destination, and dangerous file types (tar-slip).
            try:
                tar.extractall(path="/tmp/extracted", filter="data")
            except TypeError:
                # Older Python without the filter argument: validate members manually
                base = os.path.realpath("/tmp/extracted")
                for member in tar.getmembers():
                    member_path = os.path.realpath(os.path.join(base, member.name))
                    if not member_path.startswith(base + os.sep):
                        raise RuntimeError(f"Unsafe path in archive: {member.name}")
                    if member.islnk() or member.issym():
                        raise RuntimeError(f"Links not allowed in archive: {member.name}")
                tar.extractall(path="/tmp/extracted")  # nosec B202 - members validated above
        
        # Find the model directory (should be samples_18806 or similar)
        extracted_base = Path("/tmp/extracted")
        model_dirs = list(extracted_base.glob("samples_*"))
        
        if not model_dirs:
            print("Error: No model directory found in extracted files")
            sys.exit(1)
        
        source_model_dir = model_dirs[0]  # Use first found directory
        print(f"Found model directory: {source_model_dir}")
        
        # Copy all model files to target directory
        files_copied = 0
        for file_path in source_model_dir.iterdir():
            if file_path.is_file():
                target_file = target_path / file_path.name
                import shutil
                shutil.copy2(file_path, target_file)
                print(f"Copied: {file_path.name}")
                files_copied += 1
        
        print(f"Successfully extracted and copied {files_copied} model files!")
        
        # Verify essential files are present
        essential_files = ["config.json", "tokenizer.json"]
        for essential_file in essential_files:
            if not (target_path / essential_file).exists():
                print(f"Warning: Essential file {essential_file} not found!")
        
        # Clean up temporary extraction directory
        import shutil
        shutil.rmtree("/tmp/extracted")
        print("Cleanup completed.")
                
    except Exception as e:
        print(f"Error extracting model files: {e}")
        sys.exit(1)

if __name__ == "__main__":
    extract_model_files()
