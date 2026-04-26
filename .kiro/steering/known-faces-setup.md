---
inclusion: manual
---

# AuraGuard AI — Known Faces Directory Setup

The Vision Engine loads face encodings and Person Profiles from `known_faces/` at startup. This guide explains how to populate it correctly.

## Directory Structure

Each person requires exactly two files with matching base names:

```
known_faces/
  hussain.jpg          # Clear, front-facing reference photo
  hussain.json         # Person_Profile JSON
  dr_ahmed.jpg
  dr_ahmed.json
  mom.jpg
  mom.json
```

## Person Profile JSON Format

```json
{
  "name": "Hussain",
  "relationship": "son",
  "background": "Software engineer living in Tampa.",
  "last_conversation": "He told you about his new job at a tech startup."
}
```

### Field Guidelines

| Field | Description | Example |
|-------|-------------|---------|
| `name` | First name as the patient knows them | `"Hussain"` |
| `relationship` | How they relate to the patient | `"son"`, `"daughter"`, `"doctor"`, `"neighbor"` |
| `background` | One sentence biography | `"Software engineer living in Tampa."` |
| `last_conversation` | Summary of last conversation (third person, past tense) | `"He told you about his new job."` |

The `background` and `last_conversation` fields are spoken directly in the voice alert, so write them in a natural, conversational tone appropriate for the patient to hear.

## Photo Requirements

- **Format**: JPEG or PNG
- **Resolution**: At least 200×200 pixels; higher is better
- **Lighting**: Well-lit, no harsh shadows on the face
- **Angle**: Front-facing preferred; slight angles (up to ~30°) are acceptable
- **Expression**: Neutral or slight smile — avoid sunglasses, hats, or heavy occlusion
- **One face per image**: The reference image should contain only the person being registered

## Adding a New Person

```bash
# 1. Copy their photo to known_faces/
cp ~/Desktop/hussain_photo.jpg known_faces/hussain.jpg

# 2. Create their profile JSON
cat > known_faces/hussain.json << 'EOF'
{
  "name": "Hussain",
  "relationship": "son",
  "background": "Software engineer living in Tampa.",
  "last_conversation": "He told you about his new job at a tech startup."
}
EOF

# 3. Verify the Vision Engine can load it (restart required)
python -c "
import face_recognition, json, os
for f in os.listdir('known_faces'):
    if f.endswith('.jpg') or f.endswith('.png'):
        img = face_recognition.load_image_file(f'known_faces/{f}')
        encs = face_recognition.face_encodings(img)
        print(f'{f}: {len(encs)} face(s) detected')
"
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `0 face(s) detected` for a photo | Photo quality too low or face too small — use a clearer, closer photo |
| Vision Engine logs `Known_Faces_Directory empty` | Ensure files are in `known_faces/` (not a subdirectory) and have matching `.jpg`/`.json` pairs |
| Wrong person identified | Add more reference photos with different angles/lighting; the `face_recognition` library averages multiple encodings |
| `json.JSONDecodeError` on startup | Validate the `.json` file — check for trailing commas or missing quotes |

## For the Hackathon Demo

Minimum viable setup: **one person** with a clear photo and complete profile JSON. The demo is most compelling when the recognized person is physically present to walk into the glasses' field of view during the presentation.

Recommended: register 2–3 people (e.g., a team member as "son/daughter", another as "doctor") to demonstrate the system recognizing different relationships and generating different voice scripts.
