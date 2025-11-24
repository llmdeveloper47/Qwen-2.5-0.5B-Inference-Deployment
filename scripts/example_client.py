#!/usr/bin/env python3
"""
Simple example client for the intent classification API.

This demonstrates how to integrate the RunPod endpoint into your application.
"""

import os
import requests
import json
from typing import List, Dict, Any


class IntentClassificationClient:
    """Client for intent classification endpoint."""
    
    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize client.
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def classify(self, prompts: List[str], timeout: int = 30) -> Dict[str, Any]:
        """
        Classify one or more prompts.
        
        Args:
            prompts: List of text prompts or single string
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with classification results
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Prepare request payload
        payload = {
            "input": {
                "prompts": prompts
            }
        }
        
        # Send request
        try:
            response = requests.post(
                f"{self.base_url}/run",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            
            response.raise_for_status()
            
            # Extract result
            result = response.json()
            
            # Handle RunPod response structure
            if "output" in result:
                return result["output"]
            elif "result" in result:
                return result["result"]
            else:
                return result
                
        except requests.Timeout:
            return {"error": f"Request timed out after {timeout}s"}
        except requests.HTTPError as e:
            return {"error": f"HTTP error: {e}"}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}
    
    def classify_single(self, prompt: str) -> Dict[str, Any]:
        """
        Classify a single prompt and return the top prediction.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Dictionary with prediction details
        """
        result = self.classify([prompt])
        
        if "error" in result:
            return result
        
        if "results" in result and len(result["results"]) > 0:
            return result["results"][0]
        
        return {"error": "No results returned"}
    
    def classify_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple prompts.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            List of classification results
        """
        result = self.classify(prompts)
        
        if "error" in result:
            return [result]
        
        return result.get("results", [])


def demo_usage():
    """Demonstrate how to use the client."""
    # Load credentials from environment
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if not endpoint_id or not api_key:
        print("Error: Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
        print("\nExample:")
        print("  export RUNPOD_ENDPOINT_ID='your-endpoint-id'")
        print("  export RUNPOD_API_KEY='your-api-key'")
        return
    
    # Create client
    client = IntentClassificationClient(endpoint_id, api_key)
    
    print("=" * 70)
    print("Intent Classification Client Demo")
    print("=" * 70)
    
    # Example 1: Single prompt
    print("\n[Example 1: Single Prompt]")
    prompt = "Book me a flight to San Francisco next Tuesday"
    print(f"Input: {prompt}")
    
    result = client.classify_single(prompt)
    
    if "error" not in result:
        print(f"\nPrediction:")
        print(f"  Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Prompt: {result['prompt']}")
    else:
        print(f"Error: {result['error']}")
    
    # Example 2: Batch classification
    print("\n" + "=" * 70)
    print("[Example 2: Batch Classification]")
    
    prompts = [
        "Add milk and bread to my shopping list",
        "Play some jazz music",
        "Set an alarm for 7 AM tomorrow",
        "What's the weather like today?",
    ]
    
    print(f"Input: {len(prompts)} prompts")
    
    results = client.classify_batch(prompts)
    
    if results and "error" not in results[0]:
        print(f"\nResults:")
        for i, result in enumerate(results):
            print(f"\n  [{i+1}] {result['prompt'][:50]}...")
            print(f"      Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {results[0].get('error', 'Unknown error')}")
    
    # Example 3: Get full probability distribution
    print("\n" + "=" * 70)
    print("[Example 3: Probability Distribution]")
    
    prompt = "Order a large pepperoni pizza for delivery"
    print(f"Input: {prompt}")
    
    result = client.classify_single(prompt)
    
    if "error" not in result:
        probs = result['probabilities']
        print(f"\nTop 5 Classes:")
        
        # Get top 5 predictions
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:5]
        
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. Class {idx}: {probs[idx]:.4f}")
    
    print("\n" + "=" * 70)


# Example: Integration with your application
def example_integration():
    """Example of how to integrate into an application."""
    
    # Initialize client once (e.g., at application startup)
    client = IntentClassificationClient(
        endpoint_id=os.getenv("RUNPOD_ENDPOINT_ID"),
        api_key=os.getenv("RUNPOD_API_KEY")
    )
    
    # Example: Handle user query
    user_query = "Book me a flight to New York"
    
    # Classify intent
    result = client.classify_single(user_query)
    
    if "error" in result:
        # Handle error
        print(f"Classification failed: {result['error']}")
        return None
    
    # Extract predicted intent
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Use intent in your application logic
    if confidence > 0.8:
        # High confidence - proceed with action
        print(f"Intent: {predicted_class} (high confidence)")
        # route_to_booking_service(user_query)
    else:
        # Low confidence - ask for clarification
        print(f"Intent unclear (confidence: {confidence:.2f})")
        # ask_for_clarification(user_query)
    
    return predicted_class


if __name__ == "__main__":
    demo_usage()

