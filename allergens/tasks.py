from celery import shared_task
import time 
from celery.result import AsyncResult

@shared_task
def background_task_1():
    # Simulate a long-running task
    #time.sleep(3)
    print("Task 11145 is running")
    # your logic here

@shared_task
def background_task_2():
    # Another long-running task
    time.sleep(7)
    print("Task 2 is running")
    # your logic here
    

def check_task_status(request, task_id):
    result = AsyncResult(task_id)
    print({'status': result.status, 'result': result.result})

