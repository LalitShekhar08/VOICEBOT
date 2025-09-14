import speech_recognition as sr
import pyttsx3
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from typing import List, Dict
import json
import time

class VoiceRAGBot:
    def __init__(self, openai_api_key: str = None):
        """Initialize the Voice RAG Bot"""
        
        # Initialize speech recognition and synthesis
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
        
        # Initialize sentence transformer for embeddings
        print("Loading sentence transformer model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize knowledge base
        self.knowledge_base: List[str] = []
        self.knowledge_embeddings = None
        self.faiss_index = None
        
        # OpenAI API key (optional - for enhanced responses)
        self.openai_api_key = openai_api_key
        if openai_api_key:
            import openai
            openai.api_key = openai_api_key
        
        print("Voice RAG Bot initialized successfully!")
    
    def speak(self, text: str):
        """Convert text to speech"""
        print(f"Bot: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen(self) -> str:
        """Listen to user input via microphone"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Listening...")
        try:
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("Processing speech...")
            # Use Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
            
        except sr.UnknownValueError:
            error_msg = "Sorry, I couldn't understand what you said."
            print(error_msg)
            return ""
        except sr.RequestError as e:
            error_msg = f"Could not request results; {e}"
            print(error_msg)
            return ""
        except sr.WaitTimeoutError:
            print("Listening timeout - no speech detected")
            return ""
    
    def add_knowledge(self, documents: List[str]):
        """Add documents to the knowledge base"""
        print("Adding documents to knowledge base...")
        
        self.knowledge_base.extend(documents)
        
        # Create embeddings for all documents
        all_embeddings = self.encoder.encode(self.knowledge_base)
        self.knowledge_embeddings = np.array(all_embeddings).astype('float32')
        
        # Create FAISS index
        dimension = self.knowledge_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.knowledge_embeddings)
        self.faiss_index.add(self.knowledge_embeddings)
        
        print(f"Added {len(documents)} documents to knowledge base")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant documents based on query"""
        if self.faiss_index is None or len(self.knowledge_base) == 0:
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search for similar documents
        scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.knowledge_base)))
        
        # Return relevant documents
        relevant_docs = []
        for i, score in zip(indices[0], scores[0]):
            if score > 0.3:  # Similarity threshold
                relevant_docs.append(self.knowledge_base[i])
        
        return relevant_docs
    
    def generate_response(self, query: str, relevant_docs: List[str]) -> str:
        """Generate response using RAG"""
        
        if not relevant_docs:
            return "I don't have information about that topic. Could you ask something else?"
        
        # Simple RAG response generation
        context = "\n".join(relevant_docs[:2])  # Use top 2 most relevant docs
        
        # Basic response template
        if len(context) > 500:
            context = context[:500] + "..."
        
        response = f"Based on my knowledge: {context}"
        
        # If OpenAI API is available, use it for better responses
        if self.openai_api_key:
            try:
                import openai
                prompt = f"""
                Context: {context}
                
                Question: {query}
                
                Please provide a helpful and concise answer based on the context provided.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"OpenAI API error: {e}")
                # Fall back to simple response
        
        return response
    
    def save_knowledge_base(self, filename: str = "knowledge_base.json"):
        """Save knowledge base to file"""
        with open(filename, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)
        print(f"Knowledge base saved to {filename}")
    
    def load_knowledge_base(self, filename: str = "knowledge_base.json"):
        """Load knowledge base from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                documents = json.load(f)
            self.add_knowledge(documents)
            print(f"Knowledge base loaded from {filename}")
        else:
            print(f"No existing knowledge base found at {filename}")
    
    def run(self):
        """Main conversation loop"""
        self.speak("Hello! I'm your EDITH Bot. How can I help you today?")
        self.speak("Say 'add knowledge' to add new information, 'save' to save knowledge base, or ask me any question.")
        
        while True:
            user_input = self.listen()
            
            if not user_input:
                continue
            
            # Exit conditions
            if any(word in user_input for word in ['goodbye', 'stop' ,'bye', 'exit', 'quit']):
                self.speak("Goodbye Friend ! Have a great day!")
                break
            
            # Add knowledge
            elif 'add knowledge' in user_input:
                self.speak("Please tell me the information you want to add to my knowledge base.")
                knowledge = self.listen()
                if knowledge:
                    self.add_knowledge([knowledge])
                    self.speak("Thank you! I've added that to my knowledge base.")
            
            # Save knowledge base
            elif 'save' in user_input and 'knowledge' in user_input:
                self.save_knowledge_base()
                self.speak("Knowledge base saved successfully!")
            
            # Regular query
            else:
                # Retrieve relevant documents
                relevant_docs = self.retrieve_relevant_docs(user_input)
                
                # Generate and speak response
                response = self.generate_response(user_input, relevant_docs)
                self.speak(response)

# Example usage and setup
def main():
    # Initialize bot (add your OpenAI API key if you have one)
    # bot = VoiceRAGBot(openai_api_key="your-openai-api-key-here")
    bot = VoiceRAGBot()
    
    # Load existing knowledge base if available
    bot.load_knowledge_base()
    
    # Add some sample knowledge (you can skip this if loading from file)
    sample_knowledge = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
        "RAG stands for Retrieval-Augmented Generation, which combines information retrieval with text generation.",
        "Speech recognition converts spoken words into text using various algorithms and models.",
        "FAISS is a library developed by Facebook for efficient similarity search and clustering.",
        "My name is EDITH.",
        "Current location is Gachibowli Hyderabad .",
        "ConfideoIT is a Course / Internship & Training providing company.",
        "Chinni akka husband name is Deepak Naidu.",
    ]
    
    if not bot.knowledge_base:  # Only add if no existing knowledge base
        bot.add_knowledge(sample_knowledge)
    
    # Start the conversation
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        bot.speak("Goodbye!")

if __name__ == "__main__":
    main()