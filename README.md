# Voice RAG Bot Setup Guide

A simple voice-enabled chatbot with Retrieval-Augmented Generation (RAG) capabilities built in Python.

## Features

- 🎤 **Voice Input**: Speak to the bot using your microphone
- 🔊 **Voice Output**: Bot responds with speech synthesis
- 📚 **Knowledge Base**: Add and store information for the bot to reference
- 🔍 **Smart Retrieval**: Uses semantic search to find relevant information
- 💾 **Persistent Storage**: Save and load knowledge base between sessions
- 🤖 **OpenAI Integration**: Optional GPT-3.5 integration for enhanced responses

## Quick Setup

### 1. Install Required Packages

Open terminal/command prompt in VS Code and run:

```bash
pip install speechrecognition pyttsx3 pyaudio sentence-transformers faiss-cpu openai python-dotenv numpy typing-extensions
```

**Note for Windows users**: If you have issues with `pyaudio`, try:
```bash
pip install pipwin
pipwin install pyaudio
```

**Note for macOS users**: You might need:
```bash
brew install portaudio
pip install pyaudio
```

### 2. Create Project Structure

Create these files in your VS Code project folder:

```
voice-rag-bot/
├── main.py                 # Main bot implementation
├── enhanced_bot.py         # Enhanced version with better features
├── .env                    # Environment configuration
├── requirements.txt        # Dependencies list
└── README.md              # This guide
```

### 3. Configure Environment (Optional)

Create a `.env` file with your settings:

```env
# Optional: Add OpenAI API key for enhanced responses
OPENAI_API_KEY=your-openai-api-key-here

# Voice settings
SPEECH_RATE=150
SPEECH_VOLUME=0.9

# RAG settings
SIMILARITY_THRESHOLD=0.3
MAX_RELEVANT_DOCS=3
```

### 4. Run the Bot

Choose one of these files to run:

**Basic Version:**
```bash
python main.py
```

**Enhanced Version (Recommended):**
```bash
python enhanced_bot.py
```

## How to Use

### Voice Commands

- **Ask Questions**: Simply speak any question about topics in the knowledge base
- **Add Knowledge**: Say "add knowledge" then speak the information you want to add
- **Save Data**: Say "save knowledge" to persist your knowledge base
- **Get Help**: Say "help" to see available commands
- **Check Status**: Say "status" to see knowledge base information
- **Exit**: Say "goodbye", "bye", "exit", or "quit" to end the conversation

### Example Conversation

```
Bot: Hello! I'm your Voice RAG Bot. How can I help you today?
You: "What is Python?"
Bot: Based on my knowledge: Python is a high-level programming language known for its simplicity and readability...

You: "Add knowledge"
Bot: Please tell me the information you want to add.
You: "FastAPI is a modern web framework for building APIs with Python"
Bot: Thank you! I've added that to my knowledge base.

You: "What is FastAPI?"
Bot: Based on my knowledge: FastAPI is a modern web framework for building APIs with Python...
```

## Troubleshooting

### Common Issues

1. **Microphone not working**
   - Check if your microphone is connected and working
   - Try adjusting microphone permissions in your OS settings
   - Run `python -m speech_recognition` to test

2. **Speech recognition errors**
   - Ensure internet connection (Google Speech Recognition requires internet)
   - Speak clearly and avoid background noise
   - Check if microphone sensitivity needs adjustment

3. **TTS (Text-to-Speech) not working**
   - On Linux, you might need: `sudo apt-get install espeak`
   - Try different voice settings in the code

4. **Import errors**
   - Make sure all packages are installed: `pip install -r requirements.txt`
   - Use a virtual environment to avoid conflicts

### Performance Tips

- The first run will be slower as it downloads the sentence transformer model
- Keep your knowledge base focused for better retrieval accuracy
- Use clear, factual statements when adding knowledge
- Save your knowledge base regularly

## Customization

### Adding Your Own Knowledge

You can pre-populate the bot with your own knowledge by modifying the `sample_knowledge` list in the code:

```python
custom_knowledge = [
    "Your custom fact 1",
    "Your custom fact 2",
    "Domain-specific information...",
]
bot.add_knowledge(custom_knowledge)
```

### Adjusting Voice Settings

Modify these parameters in the code:

```python
# Speech rate (words per minute)
self.tts_engine.setProperty('rate', 150)

# Volume (0.0 to 1.0)
self.tts_engine.setProperty('volume', 0.9)

# Voice selection (try different indices)
voices = self.tts_engine.getProperty('voices')
self.tts_engine.setProperty('voice', voices[1].id)
```

### Similarity Threshold

Adjust how strict the similarity matching is:

```python
# Lower values = more permissive matching
# Higher values = more strict matching
self.similarity_threshold = 0.3
```

## Extending the Bot

### Adding New Features

1. **File Upload**: Add functionality to load knowledge from text files
2. **Web Search**: Integrate with search APIs for real-time information
3. **Memory**: Add conversation context and memory
4. **Multi-language**: Support for different languages
5. **GUI**: Create a graphical interface

### Integration Ideas

- Connect to company databases
- Link with documentation systems
- Integrate with calendar/scheduling systems
- Add weather/news APIs
- Connect to customer support systems

## Requirements

- Python 3.7+
- Working microphone
- Internet connection (for speech recognition)
- ~2GB free space (for AI models)

## License

This project is open source and available under the MIT License.