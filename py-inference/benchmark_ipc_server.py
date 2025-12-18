import sys
import os
import asyncio
import time
import grpc
import numpy as np
import torch

# Add generated code to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../gen/python'))

import snake_pb2
import snake_pb2_grpc

# Configuration
BATCH_SIZE = 4096
MAX_WAIT_TIME = 0.001 # 1ms

class BatchManager:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_running_loop()

    async def process_batches(self):
        print("Batch processing started...")
        last_print = time.time()
        batches_processed = 0
        total_items = 0
        
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
                    # This wait_for is a potential bottleneck
                    item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch_items:
                # Simulate inference (instant)
                self.resolve_batch(batch_items)
                batches_processed += 1
                total_items += len(batch_items)
                
                now = time.time()
                if now - last_print > 1.0:
                    print(f"IPC Benchmark: {batches_processed} batches/sec | {total_items} req/sec")
                    batches_processed = 0
                    total_items = 0
                    last_print = now

    def resolve_batch(self, batch_items):
        # Create dummy responses
        # 4 snakes * 4 moves = 16 policies
        # 4 snakes = 4 values
        dummy_policy = torch.full((16,), 0.25)
        dummy_value = torch.full((4,), 0.5)
        
        for _, future in batch_items:
            if not future.done():
                future.set_result((dummy_policy, dummy_value))

class InferenceService(snake_pb2_grpc.InferenceServiceServicer):
    def __init__(self, batch_manager):
        self.batch_manager = batch_manager

    async def Predict(self, request, context):
        # Simulate data ingestion cost
        # This tests the cost of receiving bytes and viewing them as numpy array
        data_array = np.frombuffer(request.data, dtype=np.float32)
        
        future = self.batch_manager.loop.create_future()
        await self.batch_manager.queue.put((None, future))
        
        policy, value = await future
        
        return snake_pb2.BatchInferenceResponse(responses=[
            snake_pb2.InferenceResponse(
                policies=policy.tolist(),
                values=value.tolist()
            )
        ])

async def serve():
    batch_manager = BatchManager()
    processing_task = asyncio.create_task(batch_manager.process_batches())
    
    server = grpc.aio.server()
    snake_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(batch_manager), server)
    
    socket_path = 'unix:///tmp/snek.sock'
    server.add_insecure_port(socket_path)
    print(f"Fake Inference Server started on {socket_path}")
    await server.start()
    
    try:
        await server.wait_for_termination()
    finally:
        processing_task.cancel()

if __name__ == '__main__':
    asyncio.run(serve())
