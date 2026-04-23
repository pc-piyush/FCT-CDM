import queue
import threading

job_queue = queue.Queue()
results = {}

def worker():
    while True:
        job_id, func, args = job_queue.get()
        try:
            results[job_id] = func(*args)
        except Exception as e:
            results[job_id] = str(e)
        job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()


def submit_job(job_id, func, *args):
    job_queue.put((job_id, func, args))


def get_result(job_id):
    return results.get(job_id, "Running...")