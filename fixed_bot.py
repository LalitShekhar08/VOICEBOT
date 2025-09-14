import speech_recognition as sr
import pyttsx3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from typing import List, Dict
import json
import time
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FixedVoiceRAGBot:
    def __init__(self):
        """Initialize the Fixed Voice RAG Bot with working TTS"""
        
        # Load configuration from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.speech_rate = int(os.getenv('SPEECH_RATE', 120))  # Slower rate
        self.speech_volume = float(os.getenv('SPEECH_VOLUME', 0.9))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
        self.max_relevant_docs = int(os.getenv('MAX_RELEVANT_DOCS', 3))
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize TTS with proper settings
        self.tts_engine = None
        self.initialize_tts()
        
        # Initialize sentence transformer for embeddings
        print("Loading sentence transformer model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize knowledge base
        self.knowledge_base: List[str] = []
        self.knowledge_embeddings = None
        self.faiss_index = None
        
        # OpenAI setup if API key is available
        if self.openai_api_key and self.openai_api_key != "your-openai-api-key-here":
            try:
                import openai
                openai.api_key = self.openai_api_key
                self.use_openai = True
                print("OpenAI integration enabled")
            except ImportError:
                self.use_openai = False
                print("OpenAI library not installed")
        else:
            self.use_openai = False
            print("OpenAI API key not provided")
        
        print("Fixed Voice RAG Bot initialized successfully!")
    
    def initialize_tts(self):
        """Initialize TTS engine with proper settings"""
        try:
            # Initialize TTS engine
            self.tts_engine = pyttsx3.init()
            
            # Set properties with error handling
            try:
                self.tts_engine.setProperty('rate', self.speech_rate)
                self.tts_engine.setProperty('volume', self.speech_volume)
                
                # Get available voices
                voices = self.tts_engine.getProperty('voices')
                if voices and len(voices) > 0:
                    # Try to use the first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
                    print(f"Using voice: {voices[0].id}")
                    
                    # If more than one voice, try the second (often better quality)
                    if len(voices) > 1:
                        self.tts_engine.setProperty('voice', voices[1].id)
                        print(f"Switched to voice: {voices[1].id}")
                
            except Exception as e:
                print(f"Warning: Could not set TTS properties: {e}")
            
            print("TTS engine initialized successfully")
            
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            self.tts_engine = None
    
    def speak(self, text: str):
        """Convert text to speech with better handling"""
        print(f"🤖 Bot: {text}")
        
        if self.tts_engine is None:
            print("TTS engine not available")
            return
        
        try:
            # Clear any pending speech
            self.tts_engine.stop()
            
            # Split long text into shorter chunks
            max_length = 100
            if len(text) > max_length:
                sentences = text.split('. ')
                for sentence in sentences:
                    if sentence.strip():
                        if not sentence.endswith('.'):
                            sentence += '.'
                        self._speak_chunk(sentence)
                        time.sleep(0.5)  # Brief pause between sentences
            else:
                self._speak_chunk(text)
                
        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback: try to reinitialize TTS
            self.initialize_tts()
    
    def _speak_chunk(self, text_chunk: str):
        """Speak a single chunk of text"""
        try:
            self.tts_engine.say(text_chunk)
            self.tts_engine.runAndWait()
            
            # Add a small delay to ensure completion
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error speaking chunk '{text_chunk}': {e}")
    
    def test_tts(self):
        """Test the TTS system"""
        test_messages = [
            "Testing text to speech.",
            "This is a longer message to test if the full sentence is spoken correctly.",
            "Hello! I am your voice bot and I should speak this entire sentence without cutting off."
        ]
        
        for msg in test_messages:
            print(f"\nTesting: {msg}")
            self.speak(msg)
            time.sleep(1)
    
    def listen(self) -> str:
        """Listen to user input via microphone"""
        try:
            with self.microphone as source:
                print("🎤 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("🎧 Listening... (speak now)")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"👤 You said: {text}")
            return text.lower()
            
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return ""
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return ""
    
    def add_knowledge(self, documents: List[str]):
        """Add documents to the knowledge base with progress indication"""
        if not documents:
            return
            
        print(f"📚 Adding {len(documents)} documents to knowledge base...")
        
        self.knowledge_base.extend(documents)
        
        # Create embeddings for all documents
        print("🔄 Creating embeddings...")
        all_embeddings = self.encoder.encode(self.knowledge_base, show_progress_bar=True)
        self.knowledge_embeddings = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index
        dimension = self.knowledge_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.knowledge_embeddings)
        self.faiss_index.add(self.knowledge_embeddings)
        
        print(f"✅ Knowledge base now contains {len(self.knowledge_base)} documents")
    
    def retrieve_relevant_docs(self, query: str) -> List[str]:
        """Retrieve relevant documents based on query"""
        if self.faiss_index is None or len(self.knowledge_base) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        k = min(self.max_relevant_docs, len(self.knowledge_base))
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Return relevant documents with scores above threshold
        relevant_docs = []
        print(f"🔍 Similarity scores: {scores[0]}")
        
        for i, score in zip(indices[0], scores[0]):
            if score > self.similarity_threshold:
                relevant_docs.append(self.knowledge_base[i])
                print(f"  📄 Retrieved doc {i} (score: {score:.3f})")
        
        return relevant_docs
    
    def generate_response(self, query: str, relevant_docs: List[str]) -> str:
        """Generate response using RAG"""
        
        if not relevant_docs:
            return "I don't have information about that topic in my knowledge base. You can add new information by saying add knowledge."
        
        # Use the most relevant document
        context = relevant_docs[0]
        
        # Use OpenAI if available
        if self.use_openai:
            try:
                import openai
                
                prompt = f"""Based on the following context, please provide a helpful and concise answer to the question.

Context: {context}

Question: {query}

Answer:"""

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"❌ OpenAI API error: {e}")
        
        # Simple fallback response
        return f"Based on my knowledge: {context}"
    
    def save_knowledge_base(self, filename: str = "knowledge_base.json"):
        """Save knowledge base to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
            print(f"💾 Knowledge base saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving knowledge base: {e}")
            return False
    
    def load_knowledge_base(self, filename: str = "knowledge_base.json"):
        """Load knowledge base from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    documents = json.load(f)
                self.add_knowledge(documents)
                print(f"📖 Knowledge base loaded from {filename}")
                return True
            except Exception as e:
                print(f"❌ Error loading knowledge base: {e}")
                return False
        else:
            print(f"📝 No existing knowledge base found at {filename}")
            return False
    
    def run(self):
        """Main conversation loop"""
        print("\n🚀 Starting Fixed Voice RAG Bot...")
        
        # Test TTS first
        print("Testing TTS system...")
        self.speak("Hello! I am your Voice RAG Bot.")
        time.sleep(1)
        self.speak("I can answer questions based on my knowledge base.")
        time.sleep(1)
        self.speak("Say add knowledge to add new information, or ask me anything!")
        
        while True:
            try:
                user_input = self.listen()
                
                if not user_input:
                    continue
                
                # Handle commands
                if any(word in user_input for word in ['goodbye', 'bye', 'exit', 'quit']):
                    self.speak("Goodbye! It was nice talking with you!")
                    break
                
                elif 'test speech' in user_input:
                    self.speak("This is a test of my speech system. I should speak this entire sentence clearly without cutting off.")
                
                elif 'add knowledge' in user_input:
                    self.speak("Please tell me the information you want to add to my knowledge base.")
                    knowledge = self.listen()
                    if knowledge:
                        self.add_knowledge([knowledge])
                        self.speak("Thank you! I have successfully added that information to my knowledge base.")
                    else:
                        self.speak("I did not hear anything. Please try again.")
                
                elif 'save knowledge' in user_input:
                    if self.save_knowledge_base():
                        self.speak("Knowledge base saved successfully!")
                    else:
                        self.speak("Sorry, I could not save the knowledge base.")
                
                else:
                    # Regular query
                    print(f"🔍 Processing query: {user_input}")
                    relevant_docs = self.retrieve_relevant_docs(user_input)
                    response = self.generate_response(user_input, relevant_docs)
                    self.speak(response)
                    
            except KeyboardInterrupt:
                print("\n⏹️ Bot stopped by user")
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.speak("Sorry, I encountered an error. Please try again.")

def main():
    """Main function to run the bot"""
    print("🤖 Fixed Voice RAG Bot")
    print("=" * 50)
    
    # Initialize bot
    bot = FixedVoiceRAGBot()
    
    # Test TTS system first
    print("\n🔊 Testing TTS System...")
    bot.test_tts()
    
    input("\nPress Enter to continue with the bot setup...")
    
    # Load existing knowledge base
    bot.load_knowledge_base()
    
    # Add sample knowledge if knowledge base is empty
    if not bot.knowledge_base:
        print("📚 Adding sample knowledge...")
        sample_knowledge = [
            "Python is a high-level programming language known for its simplicity and readability.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
            "RAG stands for Retrieval Augmented Generation which combines information retrieval with text generation.",
            "Speech recognition technology converts spoken language into text using acoustic models.",
            "FAISS is a library for efficient similarity search and clustering of dense vectors."
            "My name is EDITH"
            "Current location is Gachibowli Hyderabad "
            "ConfideoIT is a Course / Internship & Training providing company."
            "Chinni akka husband name is Deepak Naidu"
        ]
        bot.add_knowledge(sample_knowledge)
    
    # Start the bot
    bot.run()

if __name__ == "__main__":
    main()