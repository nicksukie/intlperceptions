import pandas as pd
import requests
import time
import re
import os

# Paths for input/output
input_csv = 'twitter_data_numeric.csv'
output_csv = 'twitter_data_with_interests.csv'

# Your interest categories
interest_categories = [
    "Economy/Business",
    "Politics/Government",
    "Technology/Innovation",
    "Traditional Culture (arts, literature, food)",
    "Sports and Entertainment",
    "Modern Lifestyle",
    "Education",
    "Science/Environment",
    "Energy/Resources",
    "Transportation/Infrastructure",
    "Military/Defense",
    "Crime/Law",
    "Health & Medicine",
    "Standard of living/Infrastructure",
    "Travel/Tourism"
]

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "[YOUR_API_KEY]"

import json
import re



def parse_json_response(content):
    # If content is inside a markdown code block, extract only the JSON part
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", content)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
    else:
        json_str = content.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
    
    
    
def get_interests_and_stubbornness(sample_content, max_retries=5):
    prompt = (
        f'Given the following user content: "{sample_content}" '
        "1. Assign between 1 and 3 interests from this list: Economy/Business, Politics/Government, Technology/Innovation, Traditional Culture (arts, literature, food), Sports and Entertainment, Modern Lifestyle, Education, Science/Environment, Energy/Resources, Transportation/Infrastructure, Military/Defense, Crime/Law, Health & Medicine, Standard of living/Infrastructure, Travel/Tourism. "
        "2. Based on semantic features (vulgar language, confidence, tone, attitude), assign a stubbornness score from 1 (not stubborn) to 10 (extremely stubborn). "
        "3. You may choose fewer than 3 interests if the content does not fit well with the possible categories. "
        "Return your answer as strict JSON in this format: "
        '{"interests": [list of interests], "stubbornness_score": integer}'
    )
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "Pro/deepseek-ai/DeepSeek-V3",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    wait_times = [5, 7, 10, 15]  # seconds, increasing on each retry
    for attempt in range(max_retries):
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            # print("API Response:", content)
            result = parse_json_response(content)
            if result is not None:
                interests = ', '.join(result.get('interests', []))
                stubbornness = result.get('stubbornness_score', None)
                print(f"Parsed interests: {interests}, stubbornness: {stubbornness}")
                return interests, stubbornness
            else:
                print("JSON parse failed, falling back to regex.")
                interests_match = re.search(r'Interests:\s*(.+)', content)
                interests = interests_match.group(1).strip() if interests_match else ''
                score_match = re.search(r'Stubbornness Score:\s*(\d+)', content)
                stubbornness = int(score_match.group(1)) if score_match else None
                return interests, stubbornness
        elif response.status_code == 429 or "TPM limit" in response.text:
            wait_time = wait_times[min(attempt, len(wait_times)-1)]
            print(f"Rate limit hit, waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        else:
            print("API error:", response.text)
            return '', None
    print("Failed after multiple retries due to rate limits.")
    return '', None

# Load main input CSV
df = pd.read_csv(input_csv)

# Try to load previous output, if exists
if os.path.exists(output_csv):
    out_df = pd.read_csv(output_csv)
    print(f"Loaded {len(out_df)} previously processed users.")
else:
    out_df = df.copy()
    out_df['interests'] = ''
    out_df['stubbornness_score'] = None

# Identify which users are already processed
already_done = (
    out_df['interests'].notnull() & out_df['interests'].astype(str).str.strip().ne('')
    & out_df['stubbornness_score'].notnull()
)

# Start/resume progress!
for idx, row in df.iterrows():
    # Use a unique identifier for each user (e.g., user_id or screen_name)
    # Modify this if you use a different unique key
    user_id = row['user_id'] if 'user_id' in row else row['screen_name']
    out_row = out_df.loc[idx]

    # If already processed, skip
    if already_done.loc[idx]:
        print(f"Skipping user {user_id} (already processed)")
        continue

    interests, stubbornness = get_interests_and_stubbornness(row['sample_content'])
    out_df.at[idx, 'interests'] = interests
    out_df.at[idx, 'stubbornness_score'] = stubbornness

    # Save after each result (avoids loss if interrupted)
    out_df.to_csv(output_csv, index=False)
    time.sleep(1)  # Adjust for rate limit

print("All users processed and saved to", output_csv)