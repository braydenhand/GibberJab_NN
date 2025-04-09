"""
NN for encoding the message. should return the model where the user can input a string and recieve 
back a vector containing the latent embedding. 
"""

class Encoder():
    def encode(self, message: str) -> list[float]:
        return [message]

    def decode(self, embedding: list[float]) -> str:
        return embedding[0]