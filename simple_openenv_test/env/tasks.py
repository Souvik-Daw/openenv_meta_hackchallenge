def get_task(task_id: int) -> dict:
    if task_id == 1:
        # Easy
        return {
            "req_id": "REQ_001",
            "text": "payment failed, my credit card was declined",
            "true_dept": "billing"
        }
    elif task_id == 2:
        # Medium
        return {
            "req_id": "REQ_002",
            "text": "app not working after payment was completed",
            "true_dept": "tech"
        }
    elif task_id == 3:
        # Hard
        return {
            "req_id": "REQ_003",
            "text": "something is wrong, I need help",
            "true_dept": "general"
        }
    else:
        raise ValueError("Invalid task ID")
