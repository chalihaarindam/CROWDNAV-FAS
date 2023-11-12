from fastapi import FastAPI
from .routers import example, execute_schema, monitor_schema, monitor, adaption_options_schema, adaption_options, execute
from .connectors import KafkaConsumerMonitor
from threading import Thread
import asyncio
from aiokafka import AIOKafkaProducer
app = FastAPI()

app.include_router(example.router, prefix="/example")
app.include_router(monitor.router, prefix="/monitor")
app.include_router(monitor_schema.router)
app.include_router(execute.router)
app.include_router(execute_schema.router)
app.include_router(adaption_options.router)
app.include_router(adaption_options_schema.router)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!----"}


@app.on_event("startup")
async def startup_event():
    await asyncio.sleep(35)
    consumerThread = Thread(target=KafkaConsumerMonitor.connect)
    consumerThread.start()




