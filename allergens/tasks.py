from celery import shared_task
import time 
from celery.result import AsyncResult
from SafeChoice.celery import app  

@shared_task
def background_task_1():
    for i in range(10):
        print("Task1",i)
    print("Task 11145 is running")
    

@shared_task
def background_task_2():
    for i in range(10,20):
        print("Task2",i)
    time.sleep(7)
    print("Task 2 is running")
    
@app.task(soft_time_limit=60)
@app.task(track_started=True)
@app.task(rate_limit="10/m")
@app.task(bind=True, max_retries=3)
def send_email(self):
    try:
        # Logic to send email
        print("Task 3 is running")
    except Exception as exc:
        # Retry the task if it fails
        raise self.retry(exc=exc)
    

def check_task_status(request, task_id):
    result = AsyncResult(task_id)
    print({'status': result.status, 'result': result.result})

