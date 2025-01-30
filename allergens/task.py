from celery import shared_task
import time

@shared_task
def long_running_task():
    time.sleep(10)  # Simulate a long task (e.g., sending an email)
    return "Task Completed!"