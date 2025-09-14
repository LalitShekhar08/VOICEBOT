import speech_recognition as sr
import pyttsx3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from typing import List, Dict
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedVoiceRAGBot:
    def __init__(self):
        """Initialize the Enhanced Voice RAG Bot"""
        
        # Load configuration from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.speech_rate = int(os.getenv('SPEECH_RATE', 150))
        self.speech_volume = float(os.getenv('SPEECH_VOLUME', 0.9))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))
        self.max_relevant_docs = int(os.getenv('MAX_RELEVANT_DOCS', 3))
        
        # Initialize speech recognition and synthesis
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS
        self.tts_engine.setProperty('rate', self.speech_rate)
        self.tts_engine.setProperty('volume', self.speech_volume)
        
        # Get available voices and set a nice one
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Try to find a female voice (usually index 1)
            if len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)
        
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
        
        print("Enhanced Voice RAG Bot initialized successfully!")
    
    def speak(self, text: str):
        """Convert text to speech"""
        print(f"🤖 Bot: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
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
            return "I don't have information about that topic in my knowledge base. You can add new information by saying 'add knowledge'."
        
        context = "\n\n".join(relevant_docs[:2])
        
        # Use OpenAI if available
        if self.use_openai:
            try:
                import openai
                
                prompt = f"""Based on the following context, please provide a helpful and concise answer to the question.

Context:
{context}

Question: {query}

Answer:"""

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"❌ OpenAI API error: {e}")
        
        # Fallback to simple response
        return f"Based on my knowledge: {context[:300]}{'...' if len(context) > 300 else ''}"
    
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
    
    def show_help(self):
        """Show available commands"""
        help_text = """
🤖 Voice RAG Bot Commands:
- Ask any question related to my knowledge base
- Say 'add knowledge' to add new information
- Say 'save knowledge' to save the knowledge base
- Say 'help' to show this help message
- Say 'status' to see knowledge base stats
- Say 'goodbye', 'bye', 'exit', or 'quit' to end conversation
        """
        print(help_text)
        self.speak("I can answer questions from my knowledge base, add new knowledge, save information, or show help. What would you like to do?")
    
    def show_status(self):
        """Show current bot status"""
        status = f"Knowledge base contains {len(self.knowledge_base)} documents. "
        if self.use_openai:
            status += "OpenAI integration is enabled."
        else:
            status += "Using basic response generation."
        
        print(f"📊 Status: {status}")
        self.speak(status)
    
    def run(self):
        """Main conversation loop"""
        print("\n🚀 Starting Voice RAG Bot...")
        self.speak("Hello! I'm your Voice RAG Bot. I can answer questions based on my knowledge base.")
        self.speak("Say help to learn about available commands, or just ask me anything!")
        
        while True:
            try:
                user_input = self.listen()
                
                if not user_input:
                    continue
                
                # Handle commands
                if any(word in user_input for word in ['goodbye', 'bye', 'exit', 'quit']):
                    self.speak("Goodbye! It was nice talking with you!")
                    break
                
                elif 'help' in user_input:
                    self.show_help()
                
                elif 'status' in user_input:
                    self.show_status()
                
                elif 'add knowledge' in user_input:
                    self.speak("Please tell me the information you want to add to my knowledge base.")
                    knowledge = self.listen()
                    if knowledge:
                        self.add_knowledge([knowledge])
                        self.speak("Thank you! I've added that information to my knowledge base.")
                    else:
                        self.speak("I didn't hear anything. Please try again.")
                
                elif 'save knowledge' in user_input:
                    if self.save_knowledge_base():
                        self.speak("Knowledge base saved successfully!")
                    else:
                        self.speak("Sorry, I couldn't save the knowledge base.")
                
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
    print("🤖 Enhanced Voice RAG Bot")
    print("=" * 50)
    
    # Initialize bot
    bot = EnhancedVoiceRAGBot()
    
    # Load existing knowledge base
    bot.load_knowledge_base()
    
    # Add sample knowledge if knowledge base is empty
    if not bot.knowledge_base:
        print("📚 Adding sample knowledge...")
        sample_knowledge = [
            "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "RAG or Retrieval-Augmented Generation combines information retrieval with text generation to provide more accurate and contextual responses.",
            "Speech recognition technology converts spoken language into text using acoustic and language models.",
            "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
            "Natural Language Processing or NLP is a branch of AI that helps computers understand, interpret and manipulate human language.",
            "Vector embeddings are numerical representations of text that capture semantic meaning and enable similarity comparisons.",
        ]
        bot.add_knowledge(sample_knowledge)
    
    # Start the bot
    bot.run()

if __name__ == "__main__":
    main()