import os
import sys

from src.extractor import NvidiaDatasetExtractor

RAW_DATA_ROOT = os.path.abspath("./nvidia_dataset_raw")
OUTPUT_ROOT = os.path.abspath("./dataset_processed")

def parse_chunk_input(user_input):
    chunks = set()
    parts = user_input.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                chunks.update(range(start, end + 1))
            except ValueError:
                print(f"Invalid range format: {part}")
        else:
            try:
                chunks.add(int(part))
            except ValueError:
                print(f"Invalid number: {part}")
    return sorted(list(chunks))

def main():
    print("="*50)
    print("Multi-View Weather Dataset Extractor")
    print("="*50)
    
    extractor = NvidiaDatasetExtractor(
        raw_root=RAW_DATA_ROOT,
        output_root=OUTPUT_ROOT
    )

    available_cams = extractor.get_available_cameras()
    
    if not available_cams:
        print("No camera folders found in 'nvidia_dataset_raw/camera'")
        return

    print("Found these cameras:")
    for idx, name in enumerate(available_cams):
        print(f"  {idx + 1}. {name}")

    print("\nWhich cameras to extract?")
    print("  [A] All Available")
    print("  [S] Select specific numbers (e.g., '1,3')")
    choice = input("Choice: ").strip().lower()

    target_cams = []
    if choice == 'a':
        target_cams = available_cams
    else:
        indices = input("Enter numbers: ").split(',')
        try:
            for i in indices:
                idx = int(i.strip()) - 1
                if 0 <= idx < len(available_cams):
                    target_cams.append(available_cams[idx])
        except ValueError:
            print("Invalid input.")
            return

    if not target_cams:
        print("No valid cameras selected.")
        return

    print(f"    Selected Cameras: {len(target_cams)}")

    user_input = input("\n  Enter chunks to extract (e.g., '1' or '1-5'): ")
    selected_chunks = parse_chunk_input(user_input)
    
    if not selected_chunks:
        print("No valid chunks selected.")
        return

    try:
        frames_input = input("How many frames per video? (Default 5): ")
        frames_per_video = int(frames_input) if frames_input.strip() else 5
    except ValueError:
        print("Invalid number. Defaulting to 5.")
        frames_per_video = 5

    print(f"\nConfiguration: {len(selected_chunks)} chunks | {len(target_cams)} cameras | {frames_per_video} imgs/video")
    if input("Start extraction? (y/n): ").lower() != 'y':
        return

    success_count = 0
    for chunk_id in selected_chunks:
        result = extractor.extract_chunk(chunk_id, target_cameras=target_cams, max_frames=frames_per_video)
        
        if result and any(result.values()):
            success_count += 1

    print("\n" + "="*50)
    print(f"    Batch Complete. Processed {success_count} chunks.")
    print("="*50)

if __name__ == "__main__":
    main()
