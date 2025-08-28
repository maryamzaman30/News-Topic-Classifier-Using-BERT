"""
Utility functions for the BERT-based news topic classifier.
This module contains functions for model loading, prediction, and text preprocessing.
"""

import os

# Try to import optional dependencies
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Try to import torch and transformers, handle gracefully if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TORCH_AVAILABLE = False

class NewsClassifier:
    """
    A wrapper class for the fine-tuned BERT news classifier.
    Handles model loading, tokenization, and prediction.
    """
    
    def __init__(self, model_dir="./fine_tuned_bert_agnews"):
        """
        Initialize the news classifier.
        
        Args:
            model_dir (str): Path to the directory containing the fine-tuned model
        """
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.metadata = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.is_loaded = False
    
    def load_model(self):
        """Load the fine-tuned model, tokenizer, and metadata."""
        try:
            if not TORCH_AVAILABLE:
                print("PyTorch and Transformers not available. Please install: pip install torch transformers")
                return False
            
            # Check if model directory exists
            if not os.path.exists(self.model_dir):
                raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'metadata.joblib')
            if os.path.exists(metadata_path) and JOBLIB_AVAILABLE:
                self.metadata = joblib.load(metadata_path)
            else:
                # Default metadata if file doesn't exist or joblib not available
                self.metadata = {
                    'class_names': ['World', 'Sports', 'Business', 'Science/Technology'],
                    'num_classes': 4,
                    'max_length': 128,
                    'model_name': 'bert-base-uncased'
                }
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text, return_probabilities=False):
        """
        Predict the class of a given text.
        
        Args:
            text (str): Input text to classify
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            dict: Prediction results containing class, confidence, and optionally all probabilities
        """
        if not TORCH_AVAILABLE:
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'error': 'PyTorch and Transformers not available. Please run the training notebook first.'
            }
        
        if not self.is_loaded:
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'error': 'Model not loaded. Please run the training notebook first.'
            }
        
        if not text or not text.strip():
            return {
                'predicted_class': 'Unknown',
                'confidence': 0.0,
                'error': 'Empty input text'
            }
        
        try:
            # Tokenize the input text
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.metadata['max_length'],
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Move tensors to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class_idx = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class_idx].item()
            predicted_class = self.metadata['class_names'][predicted_class_idx]
            
            result = {
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence
            }
            
            # Add all probabilities if requested
            if return_probabilities:
                if NUMPY_AVAILABLE:
                    all_probs = predictions[0].cpu().numpy()
                    result['all_probabilities'] = {
                        self.metadata['class_names'][i]: float(prob) 
                        for i, prob in enumerate(all_probs)
                    }
                else:
                    # Fallback without numpy
                    all_probs = predictions[0].cpu().tolist()
                    result['all_probabilities'] = {
                        self.metadata['class_names'][i]: float(prob) 
                        for i, prob in enumerate(all_probs)
                    }
            
            return result
            
        except Exception as e:
            return {
                'predicted_class': 'Error',
                'confidence': 0.0,
                'error': f'Prediction error: {str(e)}'
            }
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict classes for multiple texts.
        
        Args:
            texts (list): List of input texts to classify
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            list: List of prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return [self.predict(text, return_probabilities) for text in texts]
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including metadata and performance metrics
        """
        if not TORCH_AVAILABLE:
            return {
                "error": "PyTorch and Transformers not available",
                "class_names": ['World', 'Sports', 'Business', 'Science/Technology'],
                "num_classes": 4,
                "device": "cpu",
                "status": "Dependencies not installed"
            }
        
        if not self.is_loaded:
            return {
                "error": "Model not loaded - please run training notebook first",
                "class_names": ['World', 'Sports', 'Business', 'Science/Technology'],
                "num_classes": 4,
                "device": str(self.device),
                "status": "Model not trained"
            }
        
        info = {
            "model_directory": self.model_dir,
            "device": str(self.device),
            "class_names": self.metadata['class_names'],
            "num_classes": self.metadata['num_classes'],
            "max_length": self.metadata['max_length'],
            "base_model": self.metadata['model_name']
        }
        
        # Add performance metrics if available
        if 'test_accuracy' in self.metadata:
            info['test_accuracy'] = self.metadata['test_accuracy']
        if 'test_f1_weighted' in self.metadata:
            info['test_f1_weighted'] = self.metadata['test_f1_weighted']
        if 'test_f1_macro' in self.metadata:
            info['test_f1_macro'] = self.metadata['test_f1_macro']
        
        return info

def preprocess_text(text):
    """
    Basic text preprocessing function.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def format_prediction_output(prediction_result):
    """
    Format prediction results for display.
    
    Args:
        prediction_result (dict): Prediction result from NewsClassifier.predict()
        
    Returns:
        str: Formatted output string
    """
    if 'error' in prediction_result:
        return f"Error: {prediction_result['error']}"
    
    output = f"Predicted Class: {prediction_result['predicted_class']}\n"
    output += f"Confidence: {prediction_result['confidence']:.3f}"
    
    if 'all_probabilities' in prediction_result:
        output += "\n\nAll Class Probabilities:\n"
        for class_name, prob in prediction_result['all_probabilities'].items():
            output += f"  {class_name}: {prob:.3f}\n"
    
    return output

def get_sample_headlines():
    """
    Get sample news headlines for testing.
    
    Returns:
        list: List of sample headlines with their expected categories
    """
    return [
        {
            "text": "Apple reports record quarterly earnings beating analyst expectations",
            "expected": "Business"
        },
        {
            "text": "Scientists discover new exoplanet in habitable zone of distant star",
            "expected": "Science/Technology"
        },
        {
            "text": "Manchester United defeats Liverpool 3-1 in Premier League match",
            "expected": "Sports"
        },
        {
            "text": "UN Security Council discusses escalating tensions in Middle East region",
            "expected": "World"
        },
        {
            "text": "New AI breakthrough could revolutionize medical diagnosis accuracy",
            "expected": "Science/Technology"
        },
        {
            "text": "Stock markets surge following Federal Reserve interest rate decision",
            "expected": "Business"
        },
        {
            "text": "Olympic swimming championships break multiple world records",
            "expected": "Sports"
        },
        {
            "text": "Climate summit reaches historic agreement on carbon emissions",
            "expected": "World"
        }
    ]

# Example usage and testing functions
if __name__ == "__main__":
    # This section runs when the module is executed directly
    print("Testing NewsClassifier...")
    
    classifier = NewsClassifier()
    
    if classifier.load_model():
        print("Model loaded successfully!")
        
        # Test with sample headlines
        sample_headlines = get_sample_headlines()
        
        print("\nTesting with sample headlines:")
        print("=" * 50)
        
        for sample in sample_headlines[:3]:  # Test first 3 samples
            text = sample["text"]
            expected = sample["expected"]
            
            result = classifier.predict(text, return_probabilities=True)
            
            print(f"\nText: {text}")
            print(f"Expected: {expected}")
            print(f"Predicted: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if result['predicted_class'] == expected:
                print("✅ Correct prediction!")
            else:
                print("❌ Incorrect prediction")
        
        # Show model info
        print(f"\nModel Info:")
        info = classifier.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    else:
        print("Failed to load model. Make sure you've run the training notebook first.")
