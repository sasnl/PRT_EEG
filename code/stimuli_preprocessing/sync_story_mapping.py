import csv
import os
import glob

# Constants
PARTITIONED_STORY_FILE = "partitioned_story.csv"
MAPPING_FILE = "story_questions_mapping_new.csv"
STIMULI_DIR = "../../stim_normalized"  # Relative path to normalized stimuli directory

def load_partitioned_stories(file_path):
    stories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stories.append(row)
    return stories

def load_mapping_file(file_path):
    mapping = {}
    if not os.path.exists(file_path):
        return mapping
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            story_id = row['story_id'].strip()
            # Store with lowercase key for case-insensitive lookup
            story_key = story_id.lower()
            q_num = row['question_num'].strip()
            
            if story_key not in mapping:
                mapping[story_key] = {}
            mapping[story_key][q_num] = row
    return mapping

def main():
    partitioned_stories = load_partitioned_stories(PARTITIONED_STORY_FILE)
    existing_mapping = load_mapping_file(MAPPING_FILE)
    
    new_rows = []
    
    # Iterate through each story from the master list
    for story_row in partitioned_stories:
        filename = story_row['filename']
        
        if filename.endswith('_studio.wav'):
            story_id = filename.replace('_studio.wav', '')
        else:
            story_id = os.path.splitext(filename)[0]
            if story_id.endswith('_studio'):
                 story_id = story_id.replace('_studio', '')
        
        # Check if folder exists
        story_folder = os.path.join(STIMULI_DIR, story_id)
        if not os.path.isdir(story_folder):
             print(f"Warning: Story folder not found for {story_id}")
        
        # Construct story_path
        story_path_rel = f"stim_normalized/{story_id}/story/{filename}"
        story_path_abs = os.path.join(STIMULI_DIR, story_id, "story", filename)
        
        if not os.path.exists(story_path_abs):
             print(f"Warning: Story file not found at {story_path_abs}")
        
        # Process 5 questions
        for i in range(1, 6):
            q_num = str(i)
            
            # Use existing data if available (case-insensitive lookup)
            existing_row = {}
            story_key = story_id.lower()
            
            if story_key in existing_mapping and q_num in existing_mapping[story_key]:
                existing_row = existing_mapping[story_key][q_num]
            
            # Determine question audio path
            questions_dir = os.path.join(STIMULI_DIR, story_id, "questions")
            q_audio_path_rel = ""
            
            if os.path.isdir(questions_dir):
                # Search for matching file
                pattern = os.path.join(questions_dir, f"*q{i}.wav")
                matches = glob.glob(pattern)
                if matches:
                    # Use the first match
                    match_name = os.path.basename(matches[0])
                    q_audio_path_rel = f"stim_normalized/{story_id}/questions/{match_name}"
                else:
                     if 'question_audio_path' in existing_row and existing_row['question_audio_path']:
                          # Keep existing if still valid? Or just ignore? 
                          # User asked to check path. New file has old paths. 
                          # We prefer dynamically found paths. 
                          pass

            
            # Invert values: '1' -> '0', '0' -> '1'
            assigned_session_val = story_row['AssignedSession']
            if assigned_session_val == '1':
                assigned_session_val = '0' # pre
            elif assigned_session_val == '0':
                assigned_session_val = '1' # post
            
            new_row = {
                'story_id': story_id,
                'assigned_session': assigned_session_val,
                'emotion': story_row['emotion'], 
                'story_path': story_path_rel,
                'question_num': q_num,
                'question_text': existing_row.get('question_text', ''),
                'answer_options': existing_row.get('answer_options', ''),
                'correct_answer': existing_row.get('correct_answer', ''),
                'question_audio_path': q_audio_path_rel
            }
            new_rows.append(new_row)

    # Write to CSV
    fieldnames = ['story_id', 'assigned_session', 'emotion', 'story_path', 'question_num', 'question_text', 'answer_options', 'correct_answer', 'question_audio_path']
    
    with open(MAPPING_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"Successfully updated {MAPPING_FILE}")

if __name__ == "__main__":
    main()
