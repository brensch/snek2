import sys
import os
import asyncio
import time
import grpc
import torch
import numpy as np

# Add generated code to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../gen/python'))

import snake_pb2
import snake_pb2_grpc
from model import SnakeNet

# Configuration
BATCH_SIZE = 256
MAX_WAIT_TIME = 0.020 # 20ms
IN_CHANNELS = 17
BOARD_WIDTH = 11
BOARD_HEIGHT = 11

def state_to_tensor(game_state):
    # TODO: Implement actual feature extraction from game_state
    # game_state has: width, height, snakes, food, you_id, turn
    # For now, return a dummy tensor
    return torch.zeros((IN_CHANNELS, BOARD_WIDTH, BOARD_HEIGHT), dtype=torch.float32)

class BatchManager:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()

    async def process_batches(self):
        while True:
            batch_items = []
            
            # Wait for the first item
            item = await self.queue.get()
            batch_items.append(item)
            
            # Collect more items
            start_time = time.time()
            while len(batch_items) < BATCH_SIZE:
                elapsed = time.time() - start_time
                remaining = MAX_WAIT_TIME - elapsed
                if remaining <= 0:
                    break
                
                try:
                    # Wait for next item with timeout
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch_items:
                await self.run_inference(batch_items)

    async def run_inference(self, batch_items):
        print(f"Processing batch of size: {len(batch_items)}")
        inputs = [item[0] for item in batch_items]
        futures = [item[1] for item in batch_items]
        
        # Stack inputs
        batch_tensor = torch.stack(inputs).to(self.device)
        
        # Run model in executor to avoid blocking the event loop
        def _inference():
            with torch.no_grad():
                policies, values = self.model(batch_tensor)
            return policies.cpu(), values.cpu()

        policies, values = await self.loop.run_in_executor(None, _inference)
        
        # Distribute results
        for i, future in enumerate(futures):
            if not future.done():
                future.set_result((policies[i], values[i]))

class InferenceService(snake_pb2_grpc.InferenceServiceServicer):
    def __init__(self, batch_manager):
        self.batch_manager = batch_manager

    async def Predict(self, request, context):
        # Create tensor from raw bytes
        # request.data is a flat byte array
        # request.shape is the shape of a single board state (e.g. [17, 11, 11])
        data_array = np.frombuffer(request.data, dtype=np.float32).copy()
        
        # Reshape to (BatchSize, *Shape)
        shape = tuple(request.shape)
        if not shape:
            # Fallback or error if shape is missing
            # Assuming standard shape if not provided, or raise error
            # For now let's assume it's provided as per requirements
            pass
            
        tensor = torch.from_numpy(data_array).reshape(-1, *shape)
        
        futures = []
        for i in range(tensor.shape[0]):
            sub_tensor = tensor[i]
            future = self.batch_manager.loop.create_future()
            await self.batch_manager.queue.put((sub_tensor, future))
            futures.append(future)
            
        results = await asyncio.gather(*futures)
        
        responses = []
        for policy, value in results:
            responses.append(snake_pb2.InferenceResponse(
                policy=policy.tolist(),
                value=value.item()
            ))
            
        return snake_pb2.BatchInferenceResponse(responses=responses)

async def serve():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SnakeNet(in_channels=IN_CHANNELS, width=BOARD_WIDTH, height=BOARD_HEIGHT).to(device)
    model.eval()
    
    batch_manager = BatchManager(model, device)
    
    # Start the batch processing task
    processing_task = asyncio.create_task(batch_manager.process_batches())
    
    server = grpc.aio.server()
    snake_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(batch_manager), server)
    
    # Use Unix Domain Socket for faster local communication
    socket_path = 'unix:///tmp/snek.sock'
    server.add_insecure_port(socket_path)
    print(f"Inference Server started on {socket_path}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    finally:
        processing_task.cancel()
        try:
            await processing_task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    asyncio.run(serve())
