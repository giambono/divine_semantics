import json
import numpy as np


class FakeModel:
    def __init__(self, embedding_dim=1024):
        self.embedding_dim = embedding_dim  # Simulate the real model's output dimension

    def encode(self, text):
        """Simulate encoding by returning a random vector of the expected dimension."""
        return np.random.rand(self.embedding_dim)


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

def simulated_openai_create(model, messages, temperature):
    # Extract the prompt (normally this is the last user message)
    user_message = next(msg['content'] for msg in messages if msg['role'] == 'user')

    # Simulate parsing logic - this can be as sophisticated as you want.
    # For example, extract some dummy graph from a known terzina.
    extracted_graph = {
        "tercet": "Nel mezzo del cammin di nostra vita",
        "nodes": [
            {
                "id": "Narratore",
                "type": "Person",
                "properties": {
                    "role": "Protagonista",
                    "historical_identity": None,
                    "verses": ["Inferno 1:1-3"],
                    "theme": "Smarrimento esistenziale"
                }
            },
            {
                "id": "Selva Oscura",
                "type": "Location",
                "properties": {
                    "description": "Una selva fitta e oscura simbolo dello smarrimento",
                    "verses": ["Inferno 1:2"],
                    "theme": "Peccato e redenzione"
                }
            }
        ],
        "relationships": [
            {
                "source": "Narratore",
                "target": "Selva Oscura",
                "type": "LOCATED_IN",
                "properties": {
                    "verses": ["Inferno 1:2"]
                }
            }
        ]
    }

    # Convert the graph to JSON string (like OpenAI would do in response)
    simulated_response_content = json.dumps(extracted_graph, ensure_ascii=False)

    # Return mock response object
    return MockResponse(simulated_response_content)
