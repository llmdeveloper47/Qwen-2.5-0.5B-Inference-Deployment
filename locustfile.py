"""
Locust load testing configuration for intent classification endpoint.

Usage:
    # With environment variables:
    export RUNPOD_API_KEY="your-api-key"
    export RUNPOD_ENDPOINT_ID="your-endpoint-id"
    export BATCH_SIZE=8
    
    locust -f locustfile.py --headless -u 10 -r 2 --run-time 5m
    
    # Or specify on command line:
    locust -f locustfile.py \
        --host=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID \
        --headless -u 10 -r 2 --run-time 5m
"""

import os
import json
import random
import time
from locust import HttpUser, task, between, events


# Configuration from environment variables
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# Sample prompts for testing (will be randomly sampled)
SAMPLE_PROMPTS = [
    "Book me a flight to San Francisco next Tuesday",
    "Add milk and bread to my shopping list",
    "Play some jazz music on Spotify",
    "Set an alarm for 7 AM tomorrow morning",
    "What's the weather forecast for this weekend?",
    "Send an email to John about the project update",
    "Schedule a meeting with the team for Friday at 3 PM",
    "Find a good Italian restaurant nearby",
    "Order a large pepperoni pizza for delivery",
    "Turn on the living room lights",
    "Call Mom on her cell phone",
    "Remind me to take out the trash tonight",
    "What's my next calendar appointment?",
    "Navigate to the nearest gas station",
    "Translate 'Hello, how are you?' to Spanish",
    "Set the thermostat to 72 degrees",
    "Add eggs to my grocery list",
    "Play the latest episode of my favorite podcast",
    "What movies are playing near me tonight?",
    "Book a table for two at Olive Garden at 7 PM",
]


class IntentClassificationUser(HttpUser):
    """Simulates a user making classification requests."""
    
    # Wait time between requests (random between 0.5 and 2 seconds)
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Set the base URL if not already set
        if not self.host:
            self.host = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
    
    @task(weight=8)
    def classify_batch(self):
        """
        Send a batch of prompts for classification.
        This is the main task (weight=8 means 80% of requests).
        """
        # Randomly sample prompts for this request
        batch = random.sample(SAMPLE_PROMPTS, min(BATCH_SIZE, len(SAMPLE_PROMPTS)))
        
        payload = {
            "input": {
                "prompts": batch
            }
        }
        
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Name this request for tracking in Locust stats
        with self.client.post(
            "/run",
            data=json.dumps(payload),
            headers=headers,
            catch_response=True,
            name=f"classify_batch_{len(batch)}"
        ) as response:
            try:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                else:
                    result = response.json()
                    
                    # Extract output (RunPod wraps response)
                    if "output" in result:
                        output = result["output"]
                    elif "result" in result:
                        output = result["result"]
                    else:
                        output = result
                    
                    # Validate response
                    if "error" in output:
                        response.failure(f"API Error: {output['error']}")
                    elif "results" not in output:
                        response.failure("Missing 'results' in output")
                    elif len(output["results"]) != len(batch):
                        response.failure(f"Expected {len(batch)} results, got {len(output['results'])}")
                    else:
                        response.success()
                        
            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
            except Exception as e:
                response.failure(f"Error: {str(e)}")
    
    @task(weight=2)
    def classify_single(self):
        """
        Send a single prompt for classification.
        Lower weight (20% of requests) to test single-item latency.
        """
        prompt = random.choice(SAMPLE_PROMPTS)
        
        payload = {
            "input": {
                "prompts": [prompt]
            }
        }
        
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        
        with self.client.post(
            "/run",
            data=json.dumps(payload),
            headers=headers,
            catch_response=True,
            name="classify_single"
        ) as response:
            try:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                else:
                    result = response.json()
                    
                    if "output" in result:
                        output = result["output"]
                    elif "result" in result:
                        output = result["result"]
                    else:
                        output = result
                    
                    if "error" in output:
                        response.failure(f"API Error: {output['error']}")
                    else:
                        response.success()
                        
            except Exception as e:
                response.failure(f"Error: {str(e)}")


@events.init_command_line_parser.add_listener
def _(parser):
    """Add custom command-line arguments."""
    parser.add_argument("--api-key", type=str, default=RUNPOD_API_KEY,
                       help="RunPod API key")
    parser.add_argument("--endpoint-id", type=str, default=RUNPOD_ENDPOINT_ID,
                       help="RunPod endpoint ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size for requests")


@events.test_start.add_listener
def _(environment, **kwargs):
    """Called when test starts."""
    print("\n" + "=" * 70)
    print("Starting Locust Load Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  API Key: {RUNPOD_API_KEY[:10]}...")
    print(f"  Endpoint ID: {RUNPOD_ENDPOINT_ID}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Users: {environment.parsed_options.num_users}")
    print(f"  Spawn Rate: {environment.parsed_options.spawn_rate}")
    print(f"  Run Time: {environment.parsed_options.run_time}")
    print("=" * 70)


@events.test_stop.add_listener
def _(environment, **kwargs):
    """Called when test stops."""
    print("\n" + "=" * 70)
    print("Load Test Complete")
    print("=" * 70)
    
    stats = environment.stats
    
    print(f"\nOverall Statistics:")
    print(f"  Total Requests: {stats.total.num_requests}")
    print(f"  Total Failures: {stats.total.num_failures}")
    print(f"  Median Response Time: {stats.total.median_response_time}ms")
    print(f"  95th Percentile: {stats.total.get_response_time_percentile(0.95)}ms")
    print(f"  99th Percentile: {stats.total.get_response_time_percentile(0.99)}ms")
    print(f"  Requests/sec: {stats.total.total_rps:.2f}")
    print("=" * 70)


# Alternative: Run from Python script
def run_from_script():
    """Run Locust programmatically from script."""
    import locust.main
    
    sys.argv = [
        "locust",
        "-f", __file__,
        "--headless",
        "-u", "10",
        "-r", "2",
        "--run-time", "5m",
        "--html", "results/load_test_report.html"
    ]
    
    locust.main.main()


if __name__ == "__main__":
    # This allows running the file directly for debugging
    print("To run this file, use: locust -f locustfile.py")
    print("\nFor headless mode:")
    print("  locust -f locustfile.py --headless -u 10 -r 2 --run-time 5m")

